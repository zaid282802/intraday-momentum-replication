"""
WRDS TAQ pull: 3 intraday snapshots/day for ~3,100 names + SPY/QQQ, 2 years.

Snapshots per (day, symbol):
  w=1  last trade 09:55:00-10:00:00   -> P(10:00)   (first-30-min signal endpoint)
  w=2  last trade 15:25:00-15:30:00   -> P(15:30)   (entry for last half hour)
  w=3  last trade 15:55:00-16:00:30   -> P(close)   (exit + next day's prev_close)

Signal (built in analysis): P1000/prev_close - 1 (includes overnight gap, Gao et al. spec)
Target: Pclose/P1530 - 1

Checkpoints: data/mim/mim_YYYYMM.parquet (resumable, months already done are skipped).
Log: data/mim/pull.log
"""
import os, time, warnings, traceback
from pathlib import Path
import pandas as pd, psycopg2
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MIM  = ROOT / "data" / "mim"
MIM.mkdir(parents=True, exist_ok=True)
LOG  = MIM / "pull.log"

def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def connect():
    pg = (Path(os.environ["APPDATA"]) / "postgresql" / "pgpass.conf").read_text().strip()
    h,p,db,u,pw = pg.split(":",4)
    c = psycopg2.connect(host=h,port=int(p),dbname=db,user=u,password=pw,sslmode="require",
                         connect_timeout=30, keepalives=1, keepalives_idle=30,
                         keepalives_interval=10, keepalives_count=3)
    with c.cursor() as cur: cur.execute("SET statement_timeout = 480000")  # 8 min: cold columnar scans need it
    return c

# ---- universe ----
# Universe: an optional ticker list (data/universe.csv, column Ticker) plus iShares
# Russell 1000 / Midcap / 2000 ETF holdings files saved in data/etf_holdings/.
univ = set()
extra = ROOT/"data"/"universe.csv"
if extra.exists():
    top = pd.read_csv(extra)["Ticker"].dropna()
    univ |= {t for t in top if isinstance(t,str) and t.isalpha()}
for f in ["IWB_holdings.xlsx","IWR_holdings.xlsx","IWM_holdings.xlsx"]:
    p = ROOT/"data"/"etf_holdings"/f
    if p.exists():
        s = pd.read_excel(p)["Ticker"].dropna()
        univ |= {t for t in s if isinstance(t,str) and t.isalpha() and 1<=len(t)<=5}
univ |= {"SPY","QQQ"}
symbols = sorted(univ)
INLIST = "','".join(symbols)
log(f"universe: {len(symbols)} symbols")

conn = connect()

# ---- trading-day list from table catalog (2 years back) ----
days = []
for yr in [2024, 2025, 2026]:
    t = pd.read_sql(f"SELECT table_name FROM information_schema.tables "
                    f"WHERE table_schema='taqm_{yr}' AND table_name LIKE 'ctm_2%' ORDER BY 1", conn)
    days += [x.replace("ctm_","") for x in t.table_name]
days = sorted(d for d in days if len(d)==8 and d.isdigit() and d >= "20240805")  # drop year-level ctm_YYYY tables
log(f"trading days: {len(days)}  ({days[0]} -> {days[-1]})")

Q = """
SELECT DISTINCT ON (sym_root, w) sym_root, w, price FROM (
  SELECT sym_root, price, time_m,
    CASE WHEN time_m BETWEEN '09:55:00' AND '10:00:00' THEN 1
         WHEN time_m BETWEEN '15:25:00' AND '15:30:00' THEN 2
         WHEN time_m BETWEEN '15:55:00' AND '16:00:30' THEN 3 END AS w
  FROM taqm_{yr}.ctm_{d}
  WHERE sym_root IN ('{inlist}') AND (sym_suffix IS NULL OR sym_suffix='')
    AND (time_m BETWEEN '09:55:00' AND '10:00:00'
      OR time_m BETWEEN '15:25:00' AND '15:30:00'
      OR time_m BETWEEN '15:55:00' AND '16:00:30')
) x WHERE w IS NOT NULL
ORDER BY sym_root, w, time_m DESC
"""

# ---- month loop with checkpoints ----
months = sorted({d[:6] for d in days})
t_start = time.time()
done_days = 0
total_days = len(days)
for m in months:
    out = MIM / f"mim_{m}.parquet"
    mdays = [d for d in days if d[:6]==m]
    if out.exists():
        done_days += len(mdays)
        log(f"{m}: checkpoint exists, skip ({len(mdays)} days)")
        continue
    frames=[]
    for d in mdays:
        yr = d[:4]
        for attempt in (1,2,3):
            try:
                t0=time.time()
                df = pd.read_sql(Q.format(yr=yr, d=d, inlist=INLIST), conn)
                df["date"]=d
                frames.append(df)
                done_days += 1
                el=time.time()-t_start
                eta=(total_days-done_days)*(el/max(done_days,1))/3600
                log(f"{d}: {len(df)} rows {time.time()-t0:.0f}s | {done_days}/{total_days} | ETA {eta:.1f}h")
                break
            except Exception as e:
                log(f"{d}: attempt {attempt} failed: ...{str(e)[-160:]}")
                try: conn.close()
                except Exception: pass
                time.sleep(45*attempt)
                conn = connect()
    if frames:
        pd.concat(frames).to_parquet(out, index=False)
        log(f"{m}: checkpoint saved -> {out.name}")
log(f"DONE in {(time.time()-t_start)/3600:.1f}h")
