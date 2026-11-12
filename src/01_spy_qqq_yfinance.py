"""
MARKET INTRADAY MOMENTUM backtest (Gao-Han-Li-Zhou 2018 JFE spec, recent data).

Signal: first-morning return INCLUDING overnight gap = P(morning_end)/P(prev_close) - 1
Target: LAST half-hour return = P(close)/P(15:30) - 1
Trade:  at 15:30 go long if signal>0 (long-only variant sits in cash if <0);
        long-short variant shorts if signal<0. Exit at close (MOC).

Two datasets (yfinance caps):
  A) 1h bars, 730d  -> signal = first HOUR (9:30-10:30), target = 15:30-16:00 bar
     (last bar of Yahoo's 1h grid is exactly the last half hour). ~2 years.
  B) 30m bars, 60d  -> exact paper spec (first 30 min), small sample sanity check.

Costs: 1bp round trip on SPY/QQQ (spread ~0.2bp/side + MOC).
Also: conditioning on |signal| size (paper: effect stronger on big-move days).
"""
import warnings
import numpy as np, pandas as pd, yfinance as yf
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

COST_RT = 0.0001  # 1bp round trip

def load(sym, interval, period):
    df = yf.download(sym, interval=interval, period=period, auto_adjust=True,
                     progress=False, prepost=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Open","Close"])
    idx = df.index
    if idx.tz is None:
        df.index = idx.tz_localize("UTC").tz_convert("America/New_York")
    else:
        df.index = idx.tz_convert("America/New_York")
    return df

def build_days(df, bars_per_day):
    """Return per-day rows: prev_close, first_bar_close, p1530, close."""
    df = df.copy()
    df["day"] = df.index.date
    rows=[]
    days = sorted(df["day"].unique())
    prev_close = None
    for d in days:
        sub = df[df["day"]==d]
        if len(sub) < bars_per_day:        # half days / gaps
            prev_close = float(sub["Close"].iloc[-1]) if len(sub) else prev_close
            continue
        first_close = float(sub["Close"].iloc[0])
        last_open_px = float(sub["Close"].iloc[-2])   # price at start of last bar (15:30)
        close = float(sub["Close"].iloc[-1])
        if prev_close is not None:
            rows.append({"day":d, "prev_close":prev_close, "first":first_close,
                         "p1530":last_open_px, "close":close})
        prev_close = close
    out = pd.DataFrame(rows)
    out["sig"]  = out["first"]/out["prev_close"] - 1     # includes overnight gap
    out["last30"] = out["close"]/out["p1530"] - 1
    return out.dropna()

def evaluate(tbl, label):
    n=len(tbl)
    # predictive regression
    x=tbl["sig"].values; y=tbl["last30"].values
    b, a = np.polyfit(x, y, 1)
    resid = y-(a+b*x)
    se_b = np.sqrt(np.sum(resid**2)/(n-2)/np.sum((x-x.mean())**2))
    tstat = b/se_b
    r2 = 1-np.sum(resid**2)/np.sum((y-y.mean())**2)
    hit = float((np.sign(x)==np.sign(y))[y!=0].mean())*100
    # strategies
    ls  = np.sign(x)*y - COST_RT
    lo  = np.where(x>0, y, 0.0) - np.where(x>0, COST_RT, 0.0)
    bh  = y  # unconditional last-30min hold, no cost (benchmark shape)
    def stats(r, trades_mask=None):
        r=pd.Series(r)
        ann=r.mean()*252
        sh = r.mean()/r.std()*np.sqrt(252) if r.std()>0 else np.nan
        return ann*100, sh
    ann_ls, sh_ls = stats(ls)
    ann_lo, sh_lo = stats(lo)
    ann_bh, sh_bh = stats(bh)
    n_lo = int((x>0).sum())
    print(f"\n--- {label}  (N={n} days) ---")
    print(f"  regression: slope {b:+.4f} (t={tstat:+.2f})  R2 {r2*100:.2f}%   sign-hit rate {hit:.1f}%")
    print(f"  LONG-SHORT: ann ret {ann_ls:+.2f}%  Sharpe {sh_ls:+.2f}   (net {COST_RT*1e4:.0f}bp RT)")
    print(f"  LONG-ONLY : ann ret {ann_lo:+.2f}%  Sharpe {sh_lo:+.2f}   ({n_lo} trades)")
    print(f"  benchmark unconditional last-30m hold: ann {ann_bh:+.2f}%  Sharpe {sh_bh:+.2f}")
    # conditioning: big-signal days (top tercile |sig|), the paper's strong regime
    thr = np.quantile(np.abs(x), 2/3)
    m = np.abs(x)>=thr
    if m.sum()>20:
        ls_c = np.sign(x[m])*y[m]-COST_RT
        hit_c = float((np.sign(x[m])==np.sign(y[m]))[y[m]!=0].mean())*100
        annc, shc = stats(ls_c)
        # per-year scale: these fire ~1/3 of days
        print(f"  BIG-SIGNAL days only (top-tercile |signal|, n={int(m.sum())}): hit {hit_c:.1f}%  "
              f"per-trade {np.mean(ls_c)*1e4:+.1f}bp  Sharpe(when active) {shc:+.2f}")

for sym in ["SPY","QQQ"]:
    print("="*100)
    print(f"{sym}")
    print("="*100)
    # A) 1h x 730d
    h = load(sym, "1h", "730d")
    bars = h.groupby(h.index.date).size().mode().iloc[0]
    tA = build_days(h, bars_per_day=int(bars))
    evaluate(tA, f"A: 2y hourly (signal=first HOUR incl overnight, target=last 30m), {tA['day'].min()} to {tA['day'].max()}")
    # B) 30m x 60d exact spec
    m30 = load(sym, "30m", "60d")
    bars30 = m30.groupby(m30.index.date).size().mode().iloc[0]
    tB = build_days(m30, bars_per_day=int(bars30))
    evaluate(tB, f"B: 60d 30-min EXACT paper spec (first 30m), {tB['day'].min()} to {tB['day'].max()}")

print("\nNOTES:")
print("- Signal includes the overnight gap (paper spec).")
print("- Variant A approximates the paper's first-30m signal with the first hour (Yahoo 1h grid);")
print("  its last bar (15:30-16:00) matches the paper's last half hour exactly.")
print("- Half-days dropped. Costs 1bp round trip. Paper-era success rate was ~52-55% with")
print("  strength concentrated on high-volatility days; compare against the big-signal rows.")
