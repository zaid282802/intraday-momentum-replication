"""
MIM ANALYSIS: 30-min Market Intraday Momentum, 2y TAQ panel, ~3,100 names.

Spec (Gao-Han-Li-Zhou 2018):
  signal = P(10:00)/prev_close - 1   (includes overnight gap)
  target = P(close)/P(15:30) - 1     (last half hour)

Outputs:
  1. SPY / QQQ exact-spec 2-year answer (the headline question)
  2. Per-stock stats -> where does the effect concentrate:
     by ADV, price, 6-1 momentum, universe membership (R1000/R2000/SP500)
  3. Gross AND net-of-spread by liquidity bucket
"""
import warnings, json, urllib.request, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

ROOT = Path(__file__).resolve().parents[1]
MIM  = ROOT / "data" / "mim"
OUT  = ROOT / "results"
OUT.mkdir(exist_ok=True)

# ---------- 1. load panel ----------
files = sorted(MIM.glob("mim_*.parquet"))
print(f"loading {len(files)} monthly checkpoints...")
panel = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
piv = panel.pivot_table(index=["date","sym_root"], columns="w", values="price").reset_index()
piv.columns = ["date","sym","p1000","p1530","pclose"]
piv = piv.sort_values(["sym","date"])
piv["prev_close"] = piv.groupby("sym")["pclose"].shift(1)
piv["prev_date"]  = piv.groupby("sym")["date"].shift(1)
piv = piv.dropna(subset=["p1000","p1530","pclose","prev_close"])
piv["sig"]  = piv["p1000"]/piv["prev_close"] - 1
piv["tgt"]  = piv["pclose"]/piv["p1530"] - 1
# guards: unadjusted TAQ -> drop split-like overnight moves; drop stale/garbage
piv = piv[(piv["sig"].abs() < 0.20) & (piv["tgt"].abs() < 0.10)]
print(f"panel: {len(piv):,} stock-days, {piv['sym'].nunique()} names, {piv['date'].nunique()} days "
      f"({piv['date'].min()} -> {piv['date'].max()})")

def evaluate(df, label, cost_rt=0.0001):
    n=len(df)
    if n < 60: return None
    x=df["sig"].values; y=df["tgt"].values
    b,a = np.polyfit(x,y,1)
    resid=y-(a+b*x)
    se=np.sqrt(np.sum(resid**2)/(n-2)/np.sum((x-x.mean())**2))
    hit=float((np.sign(x)==np.sign(y))[y!=0].mean())*100
    ls = np.sign(x)*y - cost_rt
    return {"label":label,"n":n,"slope":b,"t":b/se,"hit%":hit,
            "ls_bps":np.mean(ls)*1e4,"ls_sharpe_daily":np.mean(ls)/np.std(ls)*np.sqrt(252) if np.std(ls)>0 else np.nan}

# ---------- 2. SPY / QQQ headline ----------
print("\n" + "="*110)
print("HEADLINE: SPY / QQQ exact-spec, 2 years (~500 days)")
print("="*110)
rows=[]
for s in ["SPY","QQQ"]:
    d = piv[piv.sym==s]
    r = evaluate(d, s, cost_rt=0.00005)   # 0.5bp RT on index ETFs
    if r: rows.append(r)
    # big-signal tercile
    thr = d["sig"].abs().quantile(2/3)
    r2 = evaluate(d[d["sig"].abs()>=thr], f"{s} big-signal", cost_rt=0.00005)
    if r2: rows.append(r2)
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------- 3. per-stock stats ----------
print("\ncomputing per-stock stats...")
def per_stock(g):
    if len(g) < 250: return None
    x=g["sig"].values; y=g["tgt"].values
    if np.std(x)==0: return None
    b,a=np.polyfit(x,y,1)
    resid=y-(a+b*x)
    se=np.sqrt(np.sum(resid**2)/(len(g)-2)/np.sum((x-x.mean())**2))
    hit=float((np.sign(x)==np.sign(y))[y!=0].mean())*100
    ls=np.sign(x)*y
    return pd.Series({"n":len(g),"slope":b,"t":b/se,"hit":hit,"ls_bps":np.mean(ls)*1e4,
                      "avg_px":g["pclose"].mean()})
ps = piv[~piv.sym.isin(["SPY","QQQ"])].groupby("sym").apply(per_stock).dropna()
print(f"per-stock stats: {len(ps)} names with >=250 obs")
print(f"cross-sectional mean slope t: {ps['t'].mean():+.2f} | share positive-slope: {(ps['slope']>0).mean()*100:.0f}% "
      f"| mean hit {ps['hit'].mean():.1f}% | mean LS gross {ps['ls_bps'].mean():+.1f}bp/day")

# ---------- 4. characteristics ----------
print("\nloading characteristics (yfinance ADV/momentum + universe files)...")
import yfinance as yf
syms = ps.index.tolist()
adv={}; mom={}
d = None
for i in range(0, len(syms), 250):
    chunk=syms[i:i+250]
    try:
        dd = yf.download(chunk, period="1y", interval="1d", auto_adjust=True,
                         progress=False, group_by="ticker", threads=True)
        for t in chunk:
            try:
                sub=dd[t].dropna(subset=["Close","Volume"])
                if len(sub)<130: continue
                c=sub["Close"]; v=sub["Volume"]
                adv[t]=float((c.tail(60)*v.tail(60)).mean())
                mom[t]=float(c.iloc[-21]/c.iloc[-126]-1)
            except Exception: pass
    except Exception: pass
ps["adv"]=pd.Series(adv); ps["mom61"]=pd.Series(mom)

ivv = set(pd.read_excel(ROOT/"data"/"etf_holdings"/"IVV_holdings.xlsx")["Ticker"].dropna())
iwb = set(pd.read_excel(ROOT/"data"/"etf_holdings"/"IWB_holdings.xlsx")["Ticker"].dropna())
iwm = set(pd.read_excel(ROOT/"data"/"etf_holdings"/"IWM_holdings.xlsx")["Ticker"].dropna())
ps["universe"]=["SP500" if s in ivv else "R1000" if s in iwb else "R2000" if s in iwm else "other" for s in ps.index]

# spread assumption by ADV bucket (round trip, bps)
def spread_rt_bp(a):
    if a!=a: return 40
    if a>=1e9: return 4
    if a>=1e8: return 10
    if a>=2e7: return 25
    if a>=5e6: return 50
    return 100
ps["spread_rt_bp"]=ps["adv"].map(spread_rt_bp)
ps["ls_net_bps"]=ps["ls_bps"]-ps["spread_rt_bp"]

def bucket_report(ps, key, buckets, label):
    print(f"\n--- by {label} ---")
    rows=[]
    for name, m in buckets:
        sub=ps[m]
        if len(sub)<10: continue
        rows.append({label:name,"names":len(sub),"mean_t":sub["t"].mean(),
                     "pos_slope%":(sub["slope"]>0).mean()*100,"hit%":sub["hit"].mean(),
                     "gross_bp/d":sub["ls_bps"].mean(),"net_bp/d":sub["ls_net_bps"].mean()})
    print(pd.DataFrame(rows).round(2).to_string(index=False))

q=ps["adv"].quantile([0.2,0.4,0.6,0.8])
bucket_report(ps,"adv",[
    ("mega (top20% ADV)", ps["adv"]>=q[0.8]),
    ("large", (ps["adv"]>=q[0.6])&(ps["adv"]<q[0.8])),
    ("mid",   (ps["adv"]>=q[0.4])&(ps["adv"]<q[0.6])),
    ("small", (ps["adv"]>=q[0.2])&(ps["adv"]<q[0.4])),
    ("micro (bottom20%)", ps["adv"]<q[0.2])],"liquidity")
bucket_report(ps,"universe",[(u, ps["universe"]==u) for u in ["SP500","R1000","R2000","other"]],"universe")
mq=ps["mom61"].quantile([0.33,0.67])
bucket_report(ps,"mom61",[
    ("momentum winners (top tercile)", ps["mom61"]>=mq[0.67]),
    ("middle", (ps["mom61"]>=mq[0.33])&(ps["mom61"]<mq[0.67])),
    ("losers (bottom tercile)", ps["mom61"]<mq[0.33])],"6-1 momentum")
pq=ps["avg_px"].quantile([0.33,0.67])
bucket_report(ps,"avg_px",[
    ("high price", ps["avg_px"]>=pq[0.67]),
    ("mid price", (ps["avg_px"]>=pq[0.33])&(ps["avg_px"]<pq[0.67])),
    ("low price", ps["avg_px"]<pq[0.33])],"price level")

ps.to_excel(OUT/"mim_per_stock.xlsx")
print(f"\nsaved -> {OUT/'mim_per_stock.xlsx'}")
print("\nTop 20 individual names by t-stat (min 250 obs):")
print(ps.sort_values("t",ascending=False).head(20)[["n","slope","t","hit","ls_bps","adv","universe"]].round(2).to_string())
