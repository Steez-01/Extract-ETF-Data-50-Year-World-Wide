"""
Global Stock Data Extractor — Multi-Country, 50-Year Daily Data
===============================================================
37 countries (expanded EU+Asia ticker lists): USA, Germany, France, UK, Spain, Netherlands, Switzerland,
Sweden, Italy, Japan, Australia, Canada, Brazil, India, South Korea,
Hong Kong, Singapore, Denmark, Norway, Finland, Poland, South Africa,
Belgium, Portugal, Austria, Ireland, Mexico, New Zealand,
China (CSI300), Taiwan (TWSE50), Turkey (BIST30),
Indonesia (IDX30), Saudi Arabia (TASI),
Malaysia (KLCI30), Thailand (SET50), Israel (TA-35), Greece (ATHEX20)

Fixes applied (X/XI/XII):
  - update_cache: REPLACE not union-merge
  - Wikipedia scraper: HK zero-padding + SEHK prefix strip + exclude list
  - Belgium/Portugal/Finland/Austria manual lists corrected
  - 5 new countries: China, Taiwan, Turkey, Indonesia, Saudi Arabia
  - China corrupted ticker removed; Taiwan dtype TypeError handled
  - Per-country exclude list blocks Wikipedia re-injection

Cache-expanded (XIV): 122 additional tickers merged from ticker_cache.json
  across CAC40/DAX40/FTSE100/IBEX35/AEX25/SMI20/OMXS30/FTSE_MIB/OBX25/
  OMX_Helsinki/NIFTY50/ASX200 for completeness

New countries (XIII — from GitHub repo):
  Malaysia (.KL / Bursa Malaysia KLCI30)
  Thailand (.BK / SET50)
  Israel (.TA / TA-35)
  Greece (.AT / FTSE/ATHEX 20)

Run modes:
  python script.py                          # all 37 countries, sequential
  python script.py --group europe           # Europe only (18 countries)
  python script.py --group americas         # Americas only (4 countries)
  python script.py --group asia             # Asia-Pacific + MiddleEast + Africa (15)
  python script.py --country Malaysia       # single country
  python script.py --country 'Saudi Arabia'
"""

import pandas as pd
import yfinance as yf
import ssl
import urllib.request
import io
import json
import os
import time
import logging
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────

START_DATE          = "1976-01-01"
END_DATE            = datetime.today().strftime("%Y-%m-%d")
OUTPUT_DIR          = "output"
CACHE_FILE          = "ticker_cache.json"
RETIRED_FILE        = "retired_tickers.json"   # replaces dead_tickers.json
MAX_WORKERS         = 1      # Keep at 1 — parallel yf.download() calls mix
                             # up each other's results (thread-safety bug).
DOWNLOAD_CHUNK      = 50
RETRY_ATTEMPTS      = 3
RETRY_DELAY         = 5
INTER_COUNTRY_DELAY = 8      # seconds to pause between countries (rate-limit guard)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ssl._create_default_https_context = ssl._create_unverified_context
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── EXCHANGE SUFFIX REGISTRY ─────────────────────────────────────────────────

EXCHANGE_SUFFIX = {
    "DE": ".DE", "FR": ".PA", "GB": ".L",  "ES": ".MC", "IT": ".MI",
    "NL": ".AS", "BE": ".BR", "PT": ".LS", "SE": ".ST", "NO": ".OL",
    "DK": ".CO", "FI": ".HE", "CH": ".SW", "AT": ".VI", "PL": ".WA",
    "CZ": ".PR", "HU": ".BD", "GR": ".AT", "US": "",    "CA": ".TO",
    "BR": ".SA", "MX": ".MX", "AR": ".BA", "JP": ".T",  "CN": ".SS",
    "HK": ".HK", "KR": ".KS", "AU": ".AX", "NZ": ".NZ", "IN": ".NS",
    "SG": ".SI", "TW": ".TW", "ZA": ".JO", "IL": ".TA", "SA": ".SR",
    "AE": ".AD", "QA": ".QA", "TR": ".IS", "ID": ".JK",
    "MY": ".KL", "TH": ".BK",
}

def get_suffix(country_code: str) -> str:
    return EXCHANGE_SUFFIX.get(country_code.upper(), "")

# ─── PERSISTENT TICKER CACHE ──────────────────────────────────────────────────

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def update_cache(index_name: str, tickers: list):
    """Write the current ticker list for this index to cache.
    Uses REPLACE (not union-merge) so tickers removed from the manual list
    are also removed from the cache on the next run, eliminating stale-ticker
    accumulation (root cause of Ireland/Mexico/Norway/Austria failures).
    """
    cache = load_cache()
    cache[index_name] = sorted(set(tickers))
    save_cache(cache)
    log.info(f"   Cache updated for {index_name}: {len(tickers)} tickers stored")
    return sorted(set(tickers))

def get_cached_tickers(index_name: str) -> list:
    return load_cache().get(index_name, [])

# ─── RETIRED TICKER ARCHIVE ───────────────────────────────────────────────────
# Replaces dead_tickers.json. Tickers are archived with country/index/reason/date
# metadata so the companion download_retired_tickers.py can retrieve their
# full historical data later.
#
# Format: {"JNPR": {"country":"USA","index":"SP500",
#                   "reason":"delisted_confirmed","first_failed":"2026-04-04"}}

def load_retired() -> dict:
    if os.path.exists(RETIRED_FILE):
        with open(RETIRED_FILE, "r") as f:
            return json.load(f)
    return {}

def save_retired(data: dict):
    with open(RETIRED_FILE, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)

def retire_tickers(tickers: list, country: str, index_name: str, reason: str):
    """Archive confirmed-gone tickers. Never overwrites existing entries."""
    if not tickers:
        return
    archive = load_retired()
    today   = datetime.today().strftime("%Y-%m-%d")
    added   = 0
    for t in tickers:
        if t not in archive:
            archive[t] = {"country": country, "index": index_name,
                          "reason": reason, "first_failed": today}
            added += 1
    if added:
        save_retired(archive)
        log.info(f"   Retired-ticker archive: +{added} new (total {len(archive)})")

def load_retired_set() -> set:
    return set(load_retired().keys())

def clean_double_suffix_from_cache():
    # Remove tickers with duplicated exchange suffix:
    #   dash-double:  RI-PA.PA   ABBN-SW.SW
    #   dot-double:   AIR.PA.DE  ASML.AS.AS
    #   SEHK-prefix:  SEHK:700.HK
    import re
    pat_dash = re.compile(r'^.+-[A-Z]{2,3}\.[A-Z]{2,3}$')
    pat_dot  = re.compile(r'^.*\.[A-Z]{2,3}\.[A-Z]{2,3}$')
    pat_sehk = re.compile(r'^SEHK:')
    cache = load_cache()
    changed = False
    for key in list(cache.keys()):
        clean = [t for t in cache[key]
                 if not pat_dash.match(t)
                 and not pat_dot.match(t)
                 and not pat_sehk.match(t)]
        removed = len(cache[key]) - len(clean)
        if removed:
            log.info(f"   Cache cleanup [{key}]: removed {removed} malformed tickers")
            cache[key] = clean
            changed = True
    if changed:
        save_cache(cache)

# ─── WIKIPEDIA SCRAPING ───────────────────────────────────────────────────────

TICKER_COLUMN_CANDIDATES = [
    "Ticker", "ticker", "Symbol", "symbol", "Code", "code",
    "Ticker symbol", "Stock symbol", "ISIN", "Mnemonic",
    "Abbreviation", "Short name", "Listing code", "EPIC",
]

def scrape_wikipedia_tickers(url: str, suffix: str, col_hints: list = None,
                             exclude: set = None) -> list:
    """Scrape ticker symbols from a Wikipedia table.
    
    exclude: optional set of full ticker strings (e.g. {'ELEC.LS','ENGI.LS'})
             that will be silently dropped even if Wikipedia returns them.
             Used to permanently block tickers that were removed from the manual
             list but which Wikipedia keeps re-injecting.
    """
    candidates = col_hints or TICKER_COLUMN_CANDIDATES
    exclude = exclude or set()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
        tables = pd.read_html(io.BytesIO(raw))
        tables_sorted = sorted(tables, key=len, reverse=True)
        for table in tables_sorted[:5]:
            for col in candidates:
                if col in table.columns:
                    raw_tickers = table[col].dropna().astype(str).tolist()
                    tickers = []
                    for t in raw_tickers:
                        t = t.strip().split(" ")[0].replace(" ", "").replace("\xa0", "")
                        # Strip exchange-prefix formats e.g. "SEHK:700" → "700"
                        if ":" in t:
                            t = t.split(":")[-1]
                        if not t or len(t) > 12:
                            continue
                        # Fix 3 (HK zero-padding):
                        # Wikipedia stores "700" but yfinance needs "0700.HK"
                        if suffix == ".HK" and t.isdigit():
                            t = t.zfill(4)
                        elif suffix != ".HK" and t.startswith("0") and t.isdigit():
                            # Skip leading-zero numeric codes for non-HK markets
                            continue
                        # Normalize dash-suffix e.g. "ABBN-SW" → "ABBN.SW"
                        if suffix:
                            dash_suffix = "-" + suffix.lstrip(".")
                            if t.endswith(dash_suffix):
                                t = t[:-len(dash_suffix)] + suffix
                        full_ticker = t + suffix if not t.endswith(suffix) else t
                        # Skip tickers explicitly excluded by the job definition
                        if full_ticker in exclude:
                            continue
                        tickers.append(full_ticker)
                    if tickers:
                        log.info(f"   Wikipedia: found {len(tickers)} tickers via column '{col}'")
                        return tickers
        log.warning(f"   Wikipedia: no matching ticker column found in {url}")
    except Exception as e:
        log.warning(f"   Wikipedia scrape failed: {e}")
    return []

# ─── MARKET CAP RANKING ───────────────────────────────────────────────────────

def rank_by_market_cap(tickers: list, country: str = "", index_name: str = "") -> list:
    log.info(f"   Ranking {len(tickers)} tickers by market cap...")
    if len(tickers) > 150:
        log.info(f"   Skipping market-cap sort (>{len(tickers)} tickers) — using list order.")
        return tickers
    cap_map = {}
    confirmed_gone = []
    for i in range(0, len(tickers), DOWNLOAD_CHUNK):
        chunk = tickers[i : i + DOWNLOAD_CHUNK]
        for t in chunk:
            try:
                fi = yf.Ticker(t).fast_info
                cap = getattr(fi, "market_cap", None)
                if cap and isinstance(cap, (int, float)) and cap > 0:
                    cap_map[t] = cap
            except Exception as e:
                # Only archive tickers Yahoo explicitly says are delisted.
                # "no timezone" is a transient rate-limit — never archive those.
                if "symbol may be delisted" in str(e).lower():
                    confirmed_gone.append(t)
            finally:
                time.sleep(0.05)
        time.sleep(0.5)
    retire_tickers(confirmed_gone, country, index_name, reason="delisted_confirmed")
    ranked = sorted(cap_map, key=cap_map.get, reverse=True)
    unranked = [t for t in tickers if t not in cap_map]
    log.info(f"   Ranked: {len(ranked)} valid | Skipped (no cap data): {len(unranked)}")
    return ranked + unranked

# ─── DOWNLOAD WITH RETRY ──────────────────────────────────────────────────────

def download_with_retry(tickers: list, start: str, end: str) -> pd.DataFrame:
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            data = yf.download(tickers, start=start, end=end,
                               auto_adjust=True, progress=False, threads=False)
            if data.empty:
                raise ValueError("yfinance returned an empty DataFrame")
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-ticker: top level = price type, second level = ticker
                if "Close" in data.columns.get_level_values(0):
                    close = data["Close"]
                else:
                    close = data.xs("Close", axis=1, level=1)
            else:
                # Single-ticker: flat columns (Open, High, Low, Close, Volume)
                if "Close" in data.columns:
                    close = data[["Close"]].rename(columns={"Close": tickers[0]})
                else:
                    close = data.iloc[:, :1].rename(columns={data.columns[0]: tickers[0]})
            return close
        except TypeError as e:
            # Fix D: yfinance occasionally returns integer 0 in a string-typed column
            # for certain tickers (e.g. 2325.TW). This crashes the whole batch.
            # Log it clearly and return whatever partial data was built before the error.
            log.warning(f"   dtype error during download (likely a numeric-zero in string "
                        f"column for one ticker): {e}")
            log.warning(f"   Attempting single-ticker fallback for affected batch...")
            # Try downloading tickers individually to isolate the bad one
            frames = []
            for t in tickers:
                try:
                    d = yf.download(t, start=start, end=end,
                                    auto_adjust=True, progress=False, threads=False)
                    if not d.empty:
                        col = d[["Close"]].rename(columns={"Close": t})
                        frames.append(col)
                except Exception:
                    pass
            if frames:
                return pd.concat(frames, axis=1)
            return pd.DataFrame()
        except Exception as e:
            err_str = str(e)
            if "Too Many Requests" in err_str or "Rate limited" in err_str or "YFRateLimitError" in err_str:
                wait = 60 * attempt
                log.warning(f"   Rate limited — waiting {wait}s before retry {attempt+1}...")
                time.sleep(wait)
            elif "401" in err_str or "Unauthorized" in err_str or "Invalid Crumb" in err_str:
                wait = 15 * attempt
                log.warning(f"   Yahoo auth error — waiting {wait}s before retry {attempt+1}...")
                time.sleep(wait)
            else:
                log.warning(f"   Download attempt {attempt} failed: {e}")
                if attempt < RETRY_ATTEMPTS:
                    time.sleep(RETRY_DELAY * attempt)
    log.error("   All download attempts failed.")
    return pd.DataFrame()

# ─── MAIN EXTRACTOR ───────────────────────────────────────────────────────────

def extract_country_data(job: dict):
    country    = job["country"]
    index_name = job["index_name"]
    suffix     = get_suffix(job.get("country_code", ""))
    log.info(f"\n{'='*60}\nPROCESSING: {country} — {index_name}")

    cached   = get_cached_tickers(index_name)
    log.info(f"   Cache: {len(cached)} previously known tickers")

    auto = []
    if job.get("url"):
        auto = scrape_wikipedia_tickers(job["url"], suffix, job.get("cols"),
                                        exclude=set(job.get("exclude", [])))

    manual   = job.get("manual", [])
    combined = list(set(cached) | set(auto) | set(manual))
    log.info(f"   Combined (cached + auto + manual): {len(combined)} unique tickers")

    # Remove retired (confirmed-delisted) tickers before ranking.
    # They are preserved in retired_tickers.json for later historical download.
    retired = load_retired_set()
    combined_clean = [t for t in combined if t not in retired]
    if len(combined_clean) < len(combined):
        log.info(f"   Retired-ticker filter: skipping {len(combined)-len(combined_clean)} "
                 f"archived tickers; {len(combined_clean)} remain for download")
    combined = combined_clean

    combined = update_cache(index_name, combined)
    ranked   = rank_by_market_cap(combined, country=country, index_name=index_name)

    log.info(f"   Downloading data for {len(ranked)} tickers ({START_DATE} → {END_DATE})...")
    close_df = download_with_retry(ranked, START_DATE, END_DATE)

    if close_df.empty:
        log.error(f"   No data returned for {index_name}. Skipping export.")
        return

    if close_df.columns.empty:
        log.error(f"   Close DataFrame has no columns for {index_name}. Skipping export.")
        return

    # Normalise to string for safe comparison (yfinance may alter suffix casing)
    col_set   = set(close_df.columns.astype(str))
    available = [t for t in ranked if str(t) in col_set]

    if not available:
        log.error(f"   No ranked tickers matched downloaded columns for {index_name}. "
                  f"Sample columns: {list(close_df.columns[:5])}")
        return

    close_df  = close_df[available]
    df_wide   = close_df.transpose()
    df_wide.columns = pd.to_datetime(df_wide.columns).strftime("%Y-%m-%d")

    filename = os.path.join(OUTPUT_DIR, f"{country}_{index_name}_{len(available)}_tickers.csv")
    df_wide.to_csv(filename)
    log.info(f"   ✓ Saved: {filename}  ({df_wide.shape[0]} tickers × {df_wide.shape[1]} days)")

# ═══════════════════════════════════════════════════════════════════════════════
# COUNTRY JOB LIBRARY — COMPLETE MANUAL TICKER LISTS
# ═══════════════════════════════════════════════════════════════════════════════

COUNTRY_JOBS = [

    # ══════════════════════════════════════════════════════════════
    # USA — S&P 500 (503 constituents)
    # ══════════════════════════════════════════════════════════════
    {
        "country": "USA",
        "country_code": "US",
        "index_name": "SP500",
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "cols": ["Symbol"],
        "manual": [
            "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","LLY","JPM",
            "V","UNH","XOM","TSLA","MA","AVGO","PG","JNJ","COST","HD","MRK","ABBV",
            "CVX","CRM","BAC","NFLX","KO","WMT","AMD","PEP","TMO","ORCL","CSCO",
            "ACN","MCD","ABT","WFC","LIN","DHR","TXN","IBM","PM","QCOM","NEE","INTU",
            "DIS","CAT","GE","RTX","SPGI","HON","VZ","AMGN","AMAT","BKNG","AXP",
            "LOW","ISRG","GS","SYK","BLK","T","PLD","MDT","VRTX","ELV","C","ADI",
            "DE","REGN","MMC","CI","GILD","SCHW","PGR","TJX","MO","BSX","CME","CB",
            "KLAC","LRCX","EOG","SO","SLB","MU","MDLZ","ETN","NOC","ITW","ICE",
            "ZTS","ADP","HCA","MMM","DUK","BDX","COP","AON","EW","APD","TGT","NSC",
            "EMR","ROP","SHW","GD","USB","PH","CTAS","GM","WELL","MCO","WM","MSI",
            "PSA","PCAR","FCX","ECL","NKE","F","CARR","D","OKE","FICO","AIG","MCK",
            "MAR","COF","APH","FDX","ORLY","SPG","FTNT","HLT","CCI","PNC","AFL",
            "NXPI","TDG","CSX","IDXX","TFC","FAST","MNST","KMB","ROST","AZO",
            "EXC","DHI","VRSK","GWW","DOW","CSGP","CTVA","CMG","A","MSCI","HUM",
            "TEL","EBAY","SRE","PAYX","NUE","OTIS","KDP","CPRT","GEHC","PCG","ALL",
            "KR","YUM","VICI","DLTR","GIS","PRU","AEP","PWR","MPWR","AME","EA",
            "ODFL","DXCM","ACGL","HSY","BIIB","O","ROK","BAX","ED","SYY","DAL",
            "HIG","STZ","LH","CTSH","WEC","SBUX","TROW","IQV","KHC","XEL",
            "CNC","PPG","KEYS","WST","TSCO","DVN","EIX","AWK","MTD","ULTA","MCHP",
            "HOLX","FIS","TTWO","CDNS","VLO","BALL","LMT","MKC","WAT","IT","HPQ",
            "FITB","MTB","ES","PPL","DTE","RMD","EFX","CHD","ZBRA","RF","DOV","LUV",
            "IFF","HRL","TRMB","EG","BR","STT","OMC","CAH","INVH","PFG",
            "STE","IP","CLX","FTV","ARE","IEX","CINF","EXPD","LVS","FSLR","HWM",
            "PKG","POOL","CTRA","TER","HAL","SWKS","CF","LYB","GLW","NTAP","AKAM",
            "TPR","JKHY","EMN","ENPH","CHRW","WRB","DPZ","AVY","MOS",
            "CPT","AIZ","ALLE","AOS","BWA","CPB","MHK","HSIC","NWSA","NWS",
            "LNC","PNR","RL","DVA","FMC","HII","IRM","L","AAL","NCLH","CCL","RCL",
            "MGM","WYNN","LW","TAP","GPS","PVH","HAS","UHS","CRL","VTRS",
            "CAG","SJM","PENN","NRG","AES","DXC","FOX","FOXA",
            "UA","UAA","AAP","BBWI","BEN","PRGO","RHI","SEE",
            "XRX","ZION","VFC","WU","RE","APA",
            # Removed confirmed-delisted/acquired (2022-2025):
            # ANSS (→Synopsys), BF-B, CTLT (→Nova), DISH (→DirecTV), DISCA/B/K (→WBD),
            # HA (→Alaska Air), HES (→Chevron), JNPR (→HPE), K (→Mars),
            # MNK, NLSN (→private), WBA (→private), WRK (→Smurfit)
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # GERMANY — DAX 40
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Germany",
        "country_code": "DE",
        "index_name": "DAX40",
        "url": "https://en.wikipedia.org/wiki/DAX",
        "cols": ["Ticker", "Symbol"],
        "manual": [
            # Core DAX 40 — verified yfinance .DE symbols (April 2026)
            "SAP.DE","SIE.DE","ALV.DE","AIR.DE","MRK.DE","DTE.DE","BMW.DE","MBG.DE",
            "BAS.DE","BAYN.DE","ADS.DE","IFX.DE","ENR.DE","RWE.DE","VNA.DE","HEN3.DE",
            "MTX.DE","EOAN.DE","HEI.DE","DBK.DE","FRE.DE","DPW.DE","HNR1.DE","LHA.DE",
            "CON.DE","VOW3.DE","PAH3.DE","P911.DE","ZAL.DE","QIA.DE","BEI.DE","DB1.DE",
            "SY1.DE","AFX.DE","1COV.DE","BOSS.DE","AIXA.DE","BC8.DE","TKA.DE","EVK.DE",
            "KGX.DE","HLAG.DE","SHL.DE","PUM.DE",
            # Current DAX constituents added / replaced in recent reshuffles
            "MUV2.DE","RHM.DE","CBK.DE","DTG.DE","G1A.DE","BNR.DE","SHL.DE",
            # MDAX & SDAX large caps with good yfinance coverage
            "DHER.DE","KWS.DE","EVD.DE","AIXA.DE","LOGN.DE","LEI.DE","NDA.DE",
            "CARL.DE","SRT3.DE","TKA.DE","NDX1.DE",
            # ── Added from ticker_cache.json (verified cache additions) ──
            "DHL.DE","FME.DE","G24.DE","MAN.DE","TAG.DE","ARND.DE","SRT.DE",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # FRANCE — CAC 40 + SBF 120 extras
    # ══════════════════════════════════════════════════════════════
    {
        "country": "France",
        "country_code": "FR",
        "index_name": "CAC40",
        "url": "https://en.wikipedia.org/wiki/CAC_40",
        "cols": ["Ticker", "Symbol"],
        "manual": [
            "MC.PA","OR.PA","TTE.PA","SAN.PA","AIR.PA","BNP.PA","KER.PA","SU.PA",
            "RI.PA","CAP.PA","EL.PA","CS.PA","ACA.PA","ORA.PA","VIE.PA","AI.PA",
            "DG.PA","EN.PA","BN.PA","SGO.PA","VIV.PA","ATO.PA","AC.PA","DSY.PA",
            "STM.PA","WLN.PA","HO.PA","RNO.PA","URW.PA","ML.PA","ENGI.PA","LR.PA",
            "ERF.PA","STLAP.PA","FTI.PA","BOL.PA","SK.PA","SOLB.PA","AMUN.PA",
            "EDF.PA","FP.PA","GLE.PA","LHN.PA","PUB.PA","RMS.PA","SAF.PA","SW.PA",
            "SOP.PA","TEP.PA","TFI.PA","UBI.PA","VLTSA.PA","CAT.PA","DBV.PA",
            "EDEN.PA","FDJ.PA","GTT.PA","HEX.PA","IPN.PA","LI.PA","NEX.PA","OVH.PA",
            # ── Added from ticker_cache.json (verified cache additions) ──
            "BVI.PA","CA.PA",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # UK — FTSE 100
    # ══════════════════════════════════════════════════════════════
    {
        "country": "UK",
        "country_code": "GB",
        "index_name": "FTSE100",
        "url": "https://en.wikipedia.org/wiki/FTSE_100_Index",
        "cols": ["Ticker", "EPIC", "Symbol"],
        "manual": [
            "SHEL.L","AZN.L","HSBA.L","ULVR.L","BP.L","RIO.L","GSK.L","LSEG.L",
            "NG.L","DGE.L","BARC.L","LLOY.L","STAN.L","BATS.L","REL.L","BA.L",
            "EXPN.L","HLN.L","IMB.L","ABF.L","FLTR.L","WPP.L","MNG.L","OCDO.L",
            "SKG.L","AUTO.L","CNA.L","NXT.L","RKT.L","SGRO.L","TSCO.L","VOD.L",
            "WTB.L","AAL.L","ADM.L","AGK.L","AHT.L","ANTO.L","AV.L","AVV.L","AW.L",
            "AZN.L","BBOX.L","BKG.L","BLND.L","BME.L","BNC.L","BNZL.L","BOO.L",
            "BRBY.L","BT.L","CCH.L","CRDA.L","CPG.L","DCC.L","DPLM.L","EZJ.L",
            "FRAS.L","GLEN.L","HIK.L","HLMA.L","HMS.L","HVPE.L","IAG.L","IHG.L",
            "III.L","INF.L","ITRK.L","ITV.L","JD.L","JET.L","KGF.L","LAND.L",
            "LMP.L","LGEN.L","LOGN.L","MKS.L","MNDI.L","MRO.L","NWG.L","NXT.L",
            "PCT.L","PFC.L","PHX.L","POLY.L","PSN.L","PSON.L","REX.L","RR.L",
            "RS1.L","RSA.L","SDR.L","SGE.L","SJP.L","SMT.L","SMWH.L","SN.L",
            "SVT.L","TW.L","UU.L","VTY.L","WEIR.L","WH.L","WHR.L","XP.L",
            # ── Added from ticker_cache.json (verified cache additions) ──
            "AAF.L","BAB.L","BEZ.L","BGEO.L","CCEP.L","CTEC.L","EDV.L","ENT.L",
            "FRES.L","GAW.L","HSX.L","HWDN.L","ICG.L","IGG.L","IMI.L","PRU.L",
            "PSH.L","RMV.L","RTO.L","SBRY.L","SMIN.L","SSE.L","STJ.L",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # SPAIN — IBEX 35
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Spain",
        "country_code": "ES",
        "index_name": "IBEX35",
        "url": "https://en.wikipedia.org/wiki/IBEX_35",
        "cols": ["Ticker", "Symbol"],
        "manual": [
            # IBEX 35 core — verified yfinance .MC symbols
            "ITX.MC","SAN.MC","BBVA.MC","IBE.MC","REP.MC","CABK.MC","TEF.MC",
            "ACS.MC","AENA.MC","CLNX.MC","FER.MC","GRF.MC","IAG.MC","MAP.MC",
            "MEL.MC","MTS.MC","NTGY.MC","RED.MC","SAB.MC","SLR.MC",
            "UNI.MC","VIS.MC","ACX.MC","AMS.MC","ANA.MC","BKT.MC",
            "CAF.MC","COL.MC","ENG.MC","FCC.MC","LOG.MC","PHM.MC",
            # SGRE.MC removed — Siemens Gamesa taken private 2023
            # Additional MSCI Spain large caps
            "MERLIN.MC","ENCE.MC","NEINOR.MC","A3M.MC","ROVI.MC",
            "ORYZON.MC","SOLARIA.MC","FLUIDRA.MC","CIE.MC","PUIG.MC",
            # ── Added from ticker_cache.json (verified cache additions) ──
            "ANE.MC","ELE.MC","IDR.MC","SCYR.MC","SGRE.MC","FDR.MC","MRL.MC",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # NETHERLANDS — AEX 25
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Netherlands",
        "country_code": "NL",
        "index_name": "AEX25",
        "url": "https://en.wikipedia.org/wiki/AEX_index",
        "cols": ["Ticker", "Symbol"],
        "manual": [
            "ASML.AS","SHELL.AS","ADYEN.AS","NN.AS","PHIA.AS","UNA.AS","WKL.AS",
            "HEIA.AS","ABN.AS","AGN.AS","AKZA.AS","ASM.AS","BESI.AS","DSM.AS",
            "EXOR.AS","IMCD.AS","INGA.AS","KPN.AS","MT.AS","RAND.AS","REN.AS",
            "SBM.AS","URW.AS","VPK.AS","WDP.AS","AALB.AS","AMG.AS","ARCAD.AS",
            "BAMNB.AS","BEKB.AS","FLOW.AS","GLPG.AS","HAL.AS","HEIJM.AS","NSI.AS",
            "OCI.AS","ORDINA.AS","OVS.AS","PHARM.AS","POST.AS","SIGNIFY.AS",
            "TOM2.AS","TKWY.AS","TOM2.AS","VAST.AS","VMOB.AS","WEX.AS",
            # ── Added from ticker_cache.json (verified cache additions) ──
            "AD.AS","ASRNL.AS","DSFIR.AS","EXO.AS","PRX.AS","UMG.AS",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # SWITZERLAND — SMI 20 + extras
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Switzerland",
        "country_code": "CH",
        "index_name": "SMI20",
        "url": "https://en.wikipedia.org/wiki/Swiss_Market_Index",
        "cols": ["Ticker", "Symbol"],
        "manual": [
            "NESN.SW","ROG.SW","NOVN.SW","UHR.SW","ZURN.SW","UBSG.SW","CFR.SW",
            "GIVN.SW","ABBN.SW","LONN.SW","ALC.SW","PGHN.SW","SREN.SW","SLHN.SW",
            "GEBN.SW","SIKA.SW","TEMN.SW","UREN.SW","SCMN.SW","BALN.SW","CSGN.SW",
            "CLTN.SW","DKSH.SW","EFGN.SW","EMMN.SW","FTON.SW","GAM.SW","HELN.SW",
            "HOLN.SW","IDIA.SW","IMPN.SW","IREN.SW","JBZG.SW","JUBN.SW","KARN.SW",
            "KNIN.SW","LECN.SW","LISP.SW","LSCC.SW","MBTN.SW","MOBN.SW","ORON.SW",
            "PATN.SW","PSPN.SW","SCHP.SW","SGSN.SW","SIGN.SW","SRAIL.SW","STMN.SW",
            "SWON.SW","SYNN.SW","TECN.SW","TIBN.SW","VPAS.SW","VZUG.SW","WKBN.SW",
            # ── Added from ticker_cache.json (verified cache additions) ──
            "ABB.SW","LOGN.SW","SOON.SW",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # SWEDEN — OMXS30 + large caps
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Sweden",
        "country_code": "SE",
        "index_name": "OMXS30",
        "url": "https://en.wikipedia.org/wiki/OMX_Stockholm_30",
        "cols": ["Ticker", "Symbol"],
        "manual": [
            "ERIC-B.ST","VOLV-B.ST","ATCO-A.ST","SWED-A.ST","SEB-A.ST","SAND.ST",
            "SKF-B.ST","SHB-A.ST","AZN.ST","ALFA.ST","ALIV-SDB.ST","ASSA-B.ST",
            "ATCO-B.ST","ATLET-B.ST","BOL.ST","CAST.ST","ELUX-B.ST","EPIROC-A.ST",
            "ESSITY-A.ST","ESSITY-B.ST","EVO.ST","GETI-B.ST","HEXA-B.ST","HM-B.ST",
            "HMS.ST","HUFV-A.ST","IAR.ST","INDU-A.ST","INV-B.ST","KINV-B.ST",
            "LATO-B.ST","LIF-B.ST","LOOMIS.ST","LVMH.ST","NIBE-B.ST","NDA-SE.ST",
            "NOKIA-SEK.ST","NWG.ST","PEAB-B.ST","SAAB-B.ST","SCA-B.ST","SCAB.ST",
            "SECUB.ST","SECU-B.ST","SKA-B.ST","SSAB-A.ST","SSAB-B.ST","SS.ST",
            "STERV.ST","STM.ST","SWED-B.ST","TELE2-B.ST","TELIA.ST","THULE.ST",
            "TOBII.ST","TREL-B.ST","VITR.ST","VOLV-A.ST","WALL-B.ST","WLKN.ST",
            # ── Added from ticker_cache.json (verified cache additions) ──
            "ABB.ST","ADDT-B.ST","EQT.ST","INDU-C.ST","INVE-B.ST","LIFCO-B.ST",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # ITALY — FTSE MIB 40
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Italy",
        "country_code": "IT",
        "index_name": "FTSE_MIB",
        "url": "https://en.wikipedia.org/wiki/FTSE_MIB",
        "cols": ["Ticker", "Symbol"],
        "manual": [
            "ISP.MI","UCG.MI","ENI.MI","ENEL.MI","G.MI","SRG.MI","PRY.MI","TIT.MI",
            "RACE.MI","MONC.MI","LDO.MI","A2A.MI","AGL.MI","ATL.MI","AMP.MI",
            "BGN.MI","BMED.MI","BPE.MI","BZU.MI","CPR.MI","CNH.MI","CNHI.MI",
            "CRG.MI","CVLT.MI","DIG.MI","DNLM.MI","ERG.MI","EXO.MI","FBK.MI",
            "FCA.MI","FCMA.MI","FNM.MI","GEO.MI","HER.MI","IREN.MI","INW.MI",
            "IVG.MI","MB.MI","MDBI.MI","MESH.MI","MONC.MI","MPS.MI","NWL.MI",
            "OVS.MI","PIRC.MI","PLT.MI","PST.MI","RINA.MI","REC.MI","SAVE.MI",
            "SFT.MI","SGM.MI","SFER.MI","SIAS.MI","STS.MI","STM.MI","SUN.MI",
            "TELO.MI","TEN.MI","TERNA.MI","TFI.MI","TOD.MI","TRN.MI","UBI.MI",
            # ── Added from ticker_cache.json (verified cache additions) ──
            "AZM.MI","BAMI.MI","BMPS.MI","DIA.MI","IG.MI","NEXI.MI","SPM.MI","STLAM.MI",
            "UNI.MI",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # JAPAN — Nikkei 225 (complete list)
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Japan",
        "country_code": "JP",
        "index_name": "NIKKEI225",
        "url": "https://en.wikipedia.org/wiki/Nikkei_225",
        "cols": ["Ticker", "Code", "Symbol"],
        "manual": [
            # Technology & Electronics
            "7203.T","6758.T","9984.T","8306.T","6861.T","4063.T","9432.T",
            "8035.T","6902.T","7267.T","4661.T","6501.T","6702.T","6752.T",
            "6753.T","6762.T","6857.T","6954.T","6971.T","6981.T","7733.T",
            "7751.T","7974.T","8697.T","9613.T","9983.T","9987.T",
            # Automotive
            "7011.T","7012.T","7013.T","7201.T","7205.T","7211.T","7261.T",
            "7269.T","7270.T","7272.T","5108.T","7282.T",
            # Finance & Insurance
            "8316.T","8411.T","8604.T","8725.T","8750.T","8766.T","8795.T",
            "8801.T","8802.T","8830.T","8931.T","3382.T",
            # Chemicals & Pharma
            "4151.T","4183.T","4188.T","4502.T","4503.T","4506.T","4507.T",
            "4523.T","4543.T","4568.T","4578.T","3407.T","4004.T","4005.T",
            "4021.T","4042.T","4043.T","4061.T","4062.T",
            # Construction & Real Estate
            "1721.T","1801.T","1802.T","1803.T","1808.T","1812.T","1925.T",
            "1928.T","5233.T","5332.T","5401.T","5411.T","5541.T",
            # Retail & Consumer
            "2502.T","2503.T","2801.T","2802.T","2871.T","2914.T","3086.T",
            "3099.T","7832.T","7911.T","9831.T","2768.T","8267.T",
            # Transport & Utilities
            "9001.T","9005.T","9007.T","9008.T","9020.T","9022.T",
            "9062.T","9064.T","9101.T","9104.T","9107.T","9202.T",
            "9501.T","9503.T","9531.T","9532.T",
            # Mining, Steel & Materials
            "5711.T","5713.T","5714.T","5802.T","5803.T","5901.T",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # AUSTRALIA — ASX 200
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Australia",
        "country_code": "AU",
        "index_name": "ASX200",
        "url": "https://en.wikipedia.org/wiki/S%26P/ASX_200",
        "cols": ["Ticker", "Code", "Symbol"],
        "manual": [
            "BHP.AX","CBA.AX","CSL.AX","NAB.AX","WBC.AX","ANZ.AX","MQG.AX",
            "WES.AX","RIO.AX","TLS.AX","FMG.AX","WOW.AX","GMG.AX","MFG.AX",
            "ALL.AX","APA.AX","AGL.AX","AMC.AX","ANN.AX","APT.AX","ARB.AX",
            "ASX.AX","BEN.AX","BLD.AX","BOQ.AX","BXB.AX","CAR.AX","CCP.AX",
            "CGF.AX","CHC.AX","COL.AX","CPU.AX","CWN.AX","DXS.AX","EBO.AX",
            "ELD.AX","EVN.AX","FLT.AX","GNC.AX","GPT.AX","HVN.AX","IAG.AX",
            "IEL.AX","IFL.AX","IGO.AX","ILU.AX","JBH.AX","LLC.AX","LYC.AX",
            "MGR.AX","MIN.AX","MPL.AX","MTS.AX","NEC.AX","NHF.AX","NIC.AX",
            "NWS.AX","NXT.AX","ORA.AX","ORG.AX","ORI.AX","OZL.AX","PLS.AX",
            "PMV.AX","PPT.AX","PRN.AX","QAN.AX","QBE.AX","REA.AX","REH.AX",
            "RHC.AX","RMD.AX","RRL.AX","S32.AX","SCG.AX","SDF.AX","SEK.AX",
            "SFR.AX","SGM.AX","SGP.AX","SHL.AX","SKC.AX","SLX.AX","SOL.AX",
            "SPK.AX","STO.AX","SUN.AX","SVW.AX","TAH.AX","TCL.AX","TWE.AX",
            "UNI.AX","VCX.AX","VEA.AX","VNT.AX","WAF.AX","WDS.AX","WHC.AX",
            "WOR.AX","WTC.AX","XRO.AX","Z1P.AX","29M.AX","360.AX","3PL.AX",
            "A2M.AX","ABC.AX","ABP.AX","AD8.AX","ALD.AX","ALQ.AX","ALU.AX",
            "AMP.AX","APE.AX","APX.AX","ARF.AX","ATM.AX","AUB.AX","AWC.AX",
            "AX1.AX","BAP.AX","BBN.AX","BGA.AX","BKW.AX","BPT.AX","BSL.AX",
            "BWP.AX","CAT.AX","CIA.AX","CMW.AX","CNU.AX","CQR.AX","CSR.AX",
            "CTD.AX","CWP.AX","DEG.AX","DHG.AX","DMP.AX","DNS.AX","DOW.AX",
            "DRR.AX","DSH.AX","DTL.AX","ELO.AX","EMR.AX","FCL.AX","FFI.AX",
            "FPH.AX","GOR.AX","GUD.AX","GWA.AX","HDN.AX","HLS.AX","HMC.AX",
            "HUB.AX","IPH.AX","IPL.AX","IVC.AX","JHG.AX","JHX.AX","KAR.AX",
            "KGN.AX","LFS.AX","LKE.AX","LNK.AX","LOV.AX","LTP.AX","LYL.AX",
            "MAH.AX","MAQ.AX","MCR.AX","MEZ.AX","MHJ.AX","MVF.AX","MYX.AX",
            "NCM.AX","NWL.AX","OFX.AX","OML.AX","PDL.AX","PEB.AX","PFP.AX",
            "PGH.AX","PLY.AX","PME.AX","PNI.AX","PSI.AX","PTM.AX","QMS.AX",
            "RAP.AX","RBL.AX","RCE.AX","RFF.AX","RFG.AX","RMC.AX","RWC.AX",
            "SIL.AX","SLR.AX","SMR.AX","SOM.AX","SPL.AX","SQ2.AX","SSM.AX",
            "SXL.AX","TGR.AX","THL.AX","TLC.AX","TNE.AX","TOP.AX","TUA.AX",
            "UMG.AX","UWL.AX","VUK.AX","WGX.AX","WNR.AX","WSA.AX","WSP.AX",
            # ── Added from ticker_cache.json (verified cache additions) ──
            "AIA.AX","AKE.AX","ALX.AX","AZJ.AX","BKL.AX","BRG.AX","CCX.AX","CHN.AX",
            "CIP.AX","CKF.AX","CLW.AX","CNI.AX","COH.AX","CRN.AX","CUV.AX","CWY.AX",
            "EDV.AX","EVT.AX","FBU.AX","GOZ.AX","KLS.AX","LTR.AX","MP1.AX","NHC.AX",
            "NSR.AX","NST.AX","NUF.AX","PBH.AX","PDN.AX","PRU.AX","QUB.AX","RMS.AX",
            "SBM.AX","SCP.AX","SGR.AX","SUL.AX","TLX.AX","TPG.AX","WEB.AX","WPR.AX",
            "ZIP.AX",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # CANADA — TSX 60 + large caps
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Canada",
        "country_code": "CA",
        "index_name": "TSX60",
        "url": "https://en.wikipedia.org/wiki/S%26P/TSX_60",
        "cols": ["Ticker", "Symbol"],
        "manual": [
            "RY.TO","TD.TO","ENB.TO","CNR.TO","CP.TO","BNS.TO","BMO.TO","CM.TO",
            "SU.TO","MFC.TO","TRI.TO","ABX.TO","BCE.TO","CNQ.TO","DOL.TO","FNV.TO",
            "GIB-A.TO","GWO.TO","IMO.TO","IFC.TO","L.TO","MRU.TO","NA.TO","NTR.TO",
            "OVV.TO","POW.TO","PPL.TO","QSR.TO","RCI-B.TO","SAP.TO","SLF.TO",
            "SHOP.TO","SNC.TO","SPB.TO","SRU-UN.TO","T.TO","TRP.TO","WCN.TO",
            "WFG.TO","WPM.TO","WSP.TO","X.TO","CCL-B.TO","CTC-A.TO","EMP-A.TO",
            "ERF.TO","FR.TO","H.TO","HBM.TO","KEY.TO","KL.TO","LUN.TO","MG.TO",
            "NGT.TO","NPI.TO","OGC.TO","PBA.TO","PD.TO","PKI.TO","REI-UN.TO",
            "RUS.TO","SJ.TO","STN.TO","TECK-B.TO","TOU.TO","TVA-B.TO","WJA.TO",
            "AEM.TO","AGI.TO","ALA.TO","ARX.TO","ATA.TO","ATD.TO","BB.TO",
            "BBD-B.TO","BEI-UN.TO","BHC.TO","BIP-UN.TO","BPY-UN.TO","BYD-UN.TO",
            "CAE.TO","CJR-B.TO","CLS.TO","CWB.TO","DII-B.TO","DRG-UN.TO",
            "EFN.TO","EIF.TO","EMP-B.TO","ERO.TO","ESI.TO","FCR-UN.TO",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # BRAZIL — Ibovespa (71 constituents — full list)
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Brazil",
        "country_code": "BR",
        "index_name": "IBOVESPA",
        "url": "https://en.wikipedia.org/wiki/Ibovespa",
        "cols": ["Ticker", "Symbol"],
        "manual": [
            "VALE3.SA","PETR4.SA","ITUB4.SA","PETR3.SA","BBDC4.SA","ABEV3.SA",
            "WEGE3.SA","BBAS3.SA","ITSA4.SA","B3SA3.SA","EQTL3.SA","RENT3.SA",
            "PRIO3.SA","RDOR3.SA","SUZB3.SA","VIVT3.SA","SBSP3.SA","LREN3.SA",
            "GGBR4.SA","TIMS3.SA","RADL3.SA","VBBR3.SA","UGPA3.SA","CMIG4.SA",
            "RAIL3.SA","CPLE6.SA","CSAN3.SA","HAPV3.SA","ELET3.SA","ELET6.SA",
            "CCRO3.SA","CPFE3.SA","CSNA3.SA","EMBR3.SA","KLBN11.SA","ENGI11.SA",
            "EGIE3.SA","ASAI3.SA","TOTS3.SA","ALOS3.SA","CYRE3.SA","MULT3.SA",
            "SOMA3.SA","CRFB3.SA","MRFG3.SA","BRFS3.SA","GOAU4.SA","COGN3.SA",
            "AZUL4.SA","CVCB3.SA","EZTC3.SA","MRVE3.SA","BEEF3.SA","SMTO3.SA",
            "YDUQ3.SA","PETZ3.SA","BHIA3.SA","DXCO3.SA","MGLU3.SA","FLRY3.SA",
            "ALPA4.SA","ARZZ3.SA","RAIZ4.SA","RECV3.SA","SLCE3.SA","VIVA3.SA",
            "TRPL4.SA","POMO4.SA","BPAC11.SA","STBP3.SA","BRAP4.SA","SMLS3.SA",
            # Additional B3 large caps
            "ALUP11.SA","AURE3.SA","AZEV3.SA","BRML3.SA","BRPR3.SA","CARD3.SA",
            "CMIN3.SA","CNTB3.SA","CURY3.SA","DIRR3.SA","ECOR3.SA","ENEV3.SA",
            "ESPA3.SA","EVEN3.SA","FESA4.SA","GGPS3.SA","GMAT3.SA","GRND3.SA",
            "HBSA3.SA","HYPE3.SA","IFCM3.SA","IGTI11.SA","IRBR3.SA","ITUB3.SA",
            "JHSF3.SA","LJQQ3.SA","LOGG3.SA","LWSA3.SA","MDIA3.SA","MILS3.SA",
            "MOVI3.SA","MYPK3.SA","NEOE3.SA","ODPV3.SA","OIBR3.SA","ONCO3.SA",
            "ORVR3.SA","PCAR3.SA","PDGR3.SA","PORT3.SA","PTBL3.SA","QUAL3.SA",
            "RAPT4.SA","RDNI3.SA","REDE3.SA","ROMI3.SA","RRRP3.SA","SBFG3.SA",
            "SMFT3.SA","TAEE11.SA","TGMA3.SA","TPVB3.SA","TRIS3.SA","TTEN3.SA",
            "TUPY3.SA","UNIP6.SA","USIM5.SA","VALE3.SA","VAMO3.SA","VIVA3.SA",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # INDIA — Nifty 50 + Nifty Next 50
    # ══════════════════════════════════════════════════════════════
    {
        "country": "India",
        "country_code": "IN",
        "index_name": "NIFTY50",
        "url": "https://en.wikipedia.org/wiki/NIFTY_50",
        "cols": ["Ticker", "Symbol"],
        "manual": [
            # Nifty 50
            "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
            "HINDUNILVR.NS","SBIN.NS","BHARTIARTL.NS","ITC.NS","LT.NS",
            "KOTAKBANK.NS","AXISBANK.NS","BAJFINANCE.NS","MARUTI.NS","TITAN.NS",
            "SUNPHARMA.NS","NESTLEIND.NS","HCLTECH.NS","ULTRACEMCO.NS","POWERGRID.NS",
            "NTPC.NS","WIPRO.NS","ONGC.NS","ASIANPAINT.NS","TECHM.NS","ADANIENT.NS",
            "ADANIPORTS.NS","BAJAJFINSV.NS","BPCL.NS","BRITANNIA.NS","CIPLA.NS",
            "COALINDIA.NS","DIVISLAB.NS","DRREDDY.NS","EICHERMOT.NS","GRASIM.NS",
            "HDFCLIFE.NS","HEROMOTOCO.NS","HINDALCO.NS","INDUSINDBK.NS","JSWSTEEL.NS",
            "M&M.NS","ONGC.NS","SBILIFE.NS","SHRIRAMFIN.NS","TATACONSUM.NS",
            "TATAMOTORS.NS","TATASTEEL.NS","TRENT.NS","UPL.NS",
            # Nifty Next 50
            "ABB.NS","ACC.NS","ADANIGREEN.NS","ADANITRANS.NS","AMBUJACEM.NS",
            "APOLLOHOSP.NS","AUROPHARMA.NS","BAJAJ-AUTO.NS","BANKBARODA.NS",
            "BERGEPAINT.NS","BOSCHLTD.NS","CANFINHOME.NS","CANBK.NS","CHOLAFIN.NS",
            "COLPAL.NS","DABUR.NS","DLF.NS","GAIL.NS","GODREJCP.NS","HAVELLS.NS",
            "ICICIPRULI.NS","IDFCFIRSTB.NS","IGL.NS","INDUSTOWER.NS","IOC.NS",
            "IRCTC.NS","JINDALSTEL.NS","JUBLFOOD.NS","LICHSGFIN.NS","LUPIN.NS",
            "MARICO.NS","MCDOWELL-N.NS","MOTHERSON.NS","MPHASIS.NS","MRF.NS",
            "NAUKRI.NS","NMDC.NS","OFSS.NS","PAGEIND.NS","PIIND.NS","PIDILITIND.NS",
            "PNBHOUSING.NS","POONAWALLA.NS","RECLTD.NS","SAIL.NS","SIEMENS.NS",
            "SRF.NS","TORNTPHARM.NS","TVSMOTOR.NS","VBL.NS","VEDL.NS","VOLTAS.NS",
            "ZYDUSLIFE.NS",
            # ── Added from ticker_cache.json (verified cache additions) ──
            "BEL.NS","ETERNAL.NS","INDIGO.NS","JIOFIN.NS","MAXHEALTH.NS",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # SOUTH KOREA — KOSPI 200 large caps
    # ══════════════════════════════════════════════════════════════
    {
        "country": "South Korea",
        "country_code": "KR",
        "index_name": "KOSPI",
        "url": "https://en.wikipedia.org/wiki/KOSPI",
        "cols": ["Ticker", "Symbol", "Code"],
        "manual": [
            "005930.KS","000660.KS","035420.KS","051910.KS","006400.KS",
            "207940.KS","035720.KS","000270.KS","068270.KS","105560.KS",
            "055550.KS","028260.KS","012330.KS","066570.KS","003550.KS",
            "010130.KS","096770.KS","017670.KS","034730.KS","003490.KS",
            "000810.KS","011200.KS","015760.KS","032640.KS","034020.KS",
            "036570.KS","047050.KS","086790.KS","138040.KS","139480.KS",
            "161390.KS","180640.KS","192820.KS","214420.KS","247540.KS",
            "251270.KS","267250.KS","271560.KS","282330.KS","316140.KS",
            "326030.KS","329180.KS","336370.KS","352820.KS","361610.KS",
            "373220.KS","402340.KS","003410.KS","004020.KS","005380.KS",
            "005490.KS","006360.KS","007070.KS","009150.KS","009830.KS",
            "010120.KS","010950.KS","011070.KS","011170.KS","011790.KS",
            "012450.KS","014820.KS","018260.KS","018880.KS","023530.KS",
            "024110.KS","028050.KS","029780.KS","030200.KS","033780.KS",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # HONG KONG — Hang Seng Index (correct yfinance format: 4-digit.HK)
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Hong Kong",
        "country_code": "HK",
        "index_name": "HSI",
        "url": "https://en.wikipedia.org/wiki/Hang_Seng_Index",
        "cols": ["Ticker", "Code", "Symbol"],
        "manual": [
            # Must use zero-padded format for yfinance
            "0700.HK","0005.HK","0941.HK","1299.HK","0388.HK","2318.HK",
            "0016.HK","1093.HK","0883.HK","2628.HK","0003.HK","0006.HK",
            "0011.HK","0012.HK","0027.HK","0066.HK","0101.HK","0175.HK",
            "0241.HK","0267.HK","0285.HK","0288.HK","0291.HK","0300.HK",
            "0316.HK","0322.HK","0386.HK","0669.HK","0688.HK","0762.HK",
            "0823.HK","0836.HK","0857.HK","0868.HK","0881.HK","0939.HK",
            "0960.HK","0968.HK","0981.HK","0992.HK","1024.HK","1038.HK",
            "1044.HK","1088.HK","1099.HK","1109.HK","1113.HK","1177.HK",
            "1209.HK","1211.HK","1299.HK","1378.HK","1398.HK","1810.HK",
            "1876.HK","1928.HK","1929.HK","1997.HK","2015.HK","2018.HK",
            "2020.HK","2057.HK","2269.HK","2313.HK","2319.HK","2331.HK",
            "2359.HK","2382.HK","2388.HK","2618.HK","2628.HK","2688.HK",
            "2899.HK","3690.HK","3692.HK","3968.HK","3988.HK","6618.HK",
            "6690.HK","6862.HK","9618.HK","9633.HK","9888.HK","9961.HK",
            "9988.HK","9992.HK","9999.HK",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # SINGAPORE — Straits Times Index + SGX large caps
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Singapore",
        "country_code": "SG",
        "index_name": "STI30",
        "url": "https://en.wikipedia.org/wiki/Straits_Times_Index",
        "cols": ["Ticker", "Symbol"],
        "manual": [
            "D05.SI","O39.SI","U11.SI","Z74.SI","C52.SI","G13.SI","BN4.SI",
            "C6L.SI","Y92.SI","V03.SI","A17U.SI","BS6.SI","C07.SI","C09.SI",
            "C38U.SI","D01.SI","F34.SI","H78.SI","J36.SI","J69U.SI","K71U.SI",
            "M44U.SI","ME8U.SI","N2IU.SI","N4E.SI","NS8U.SI","P15.SI","RE4.SI",
            "S51.SI","S58.SI","S68.SI","T82U.SI","U09.SI","U96.SI","UD2.SI",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # DENMARK — OMX Copenhagen 25
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Denmark",
        "country_code": "DK",
        "index_name": "OMXC25",
        "url": "https://en.wikipedia.org/wiki/OMX_Copenhagen_25",
        "cols": ["Ticker", "Symbol"],
        "manual": [
            "NOVO-B.CO","MAERSK-B.CO","DSV.CO","ORSTED.CO","CARL-B.CO",
            "COLO-B.CO","GN.CO","TRYG.CO","PNDORA.CO","RBREW.CO",
            "AMBU-B.CO","CHR.CO","DEMANT.CO","FLS.CO","GMAB.CO",
            "ISS.CO","JYSK.CO","NNIT.CO","NTG.CO","NZYM-B.CO",
            "ROCK-B.CO","SIM.CO","SOLAR-B.CO","SPLY.CO","VWS.CO",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # NORWAY — OBX 25 + large caps
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Norway",
        "country_code": "NO",
        "index_name": "OBX25",
        "url": "https://en.wikipedia.org/wiki/OBX_Stock_Index",
        "cols": ["Ticker", "Symbol"],
        "manual": [
            # Verified Oslo Bors (OBX / large-cap) tickers on yfinance .OL
            "EQNR.OL","DNB.OL","MOWI.OL","YAR.OL","TEL.OL","ORK.OL",
            "NHY.OL","AKRBP.OL","BAKKA.OL","SUBC.OL","SCHA.OL",
            "AKER.OL","AKSO.OL","FRO.OL","GOGL.OL","HAFNIA.OL",
            "SALM.OL","STB.OL","TGS.OL","MPCC.OL","BORR.OL",
            "FLNG.OL","COOL.OL","PGS.OL","NEL.OL","SDRL.OL",
            "RECSI.OL","GJF.OL","SRBANK.OL","ATEA.OL","BOUVET.OL",
            "ENTRA.OL","SATS.OL","OKEA.OL","AGAS.OL","BWO.OL",
            "KAHOOT.OL","CRAYON.OL","NXTECH.OL","THIN.OL","NSG.OL",
            "ASETEK.OL","INSR.OL","KIT.OL","LINK.OL","SOFF.OL",
            # ── Added from ticker_cache.json (verified cache additions) ──
            "BWLPG.OL","NSKOG.OL","ODF.OL","WAWI.OL","AFG.OL","ARCH.OL","HUNT.OL","NAVA.OL",
            "NEXT.OL",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # FINLAND — OMX Helsinki 25
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Finland",
        "country_code": "FI",
        "index_name": "OMX_Helsinki",
        "url": "https://en.wikipedia.org/wiki/OMX_Helsinki_25",
        "cols": ["Ticker", "Symbol"],
        # Wikipedia re-injects these removed tickers on every run.
        # Blocked permanently at scraper level via exclude list.
        "exclude": [
            "KONE.HE","CGCBV.HE","YIT1V.HE",
            "RTRKS.HE","TIE1V.HE","SAMAS.HE",
        ],
        "manual": [
            # Verified Nasdaq Helsinki (.HE) — April 2026
            # Removed (no yfinance data / confirmed gone):
            #   KONE.HE (→ use KNEBV.HE already present), CGCBV.HE, YIT1V.HE,
            #   RTRKS.HE, TIE1V.HE, SAMAS.HE
            "NOKIA.HE","KNEBV.HE","NESTE.HE","SAMPO.HE","OUT1V.HE","FORTUM.HE",
            "WRT1V.HE","STERV.HE","KEMIRA.HE","TIETO.HE","ELISA.HE",
            "FSKRS.HE","HUH1V.HE","ICP1V.HE","KESKOB.HE",
            "METSO.HE","METSB.HE","NDA-FI.HE","ORNBV.HE","PIHLIS.HE","QTCOM.HE",
            "RAIVV.HE","SRV1V.HE","SSABAH.HE","TEM1V.HE",
            "TNOM.HE","UPM.HE","VALMT.HE","KOJAMO.HE","LEHTO.HE",
            # ── Added from ticker_cache.json (verified cache additions) ──
            "HIAB.HE","KALMAR.HE","KCR.HE","TYRES.HE",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # POLAND — WIG 20 + mWIG 40 extras
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Poland",
        "country_code": "PL",
        "index_name": "WIG20",
        "url": "https://en.wikipedia.org/wiki/WIG20",
        "cols": ["Ticker", "Symbol"],
        "manual": [
            "PKN.WA","PKO.WA","PZU.WA","DNP.WA","JSW.WA","LPP.WA","OPL.WA",
            "CDR.WA","ALE.WA","CCC.WA","DIINO.WA","EUR.WA","HRS.WA","ING.WA",
            "KGH.WA","MBK.WA","MRC.WA","PCO.WA","PEPCO.WA","PEO.WA",
            "PKP.WA","PLW.WA","PZU.WA","SPH.WA","TEN.WA",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # SOUTH AFRICA — JSE Top 40
    # ══════════════════════════════════════════════════════════════
    {
        "country": "South Africa",
        "country_code": "ZA",
        "index_name": "JSE_TOP40",
        "url": "https://en.wikipedia.org/wiki/JSE_Securities_Exchange",
        "cols": ["Ticker", "Symbol"],
        "manual": [
            "NPN.JO","PRX.JO","BTI.JO","AGL.JO","SOL.JO","SBK.JO","NED.JO",
            "FSR.JO","ABG.JO","CFR.JO","BVT.JO","DSY.JO","GRT.JO","HAR.JO",
            "IMP.JO","INL.JO","INP.JO","IPL.JO","LHC.JO","MCG.JO","MNP.JO",
            "MRP.JO","MTN.JO","MUR.JO","NHM.JO","NRP.JO","OMU.JO","PPH.JO",
            "RBP.JO","RDF.JO","REM.JO","RNI.JO","SHP.JO","SPP.JO","TFG.JO",
            "TGA.JO","TRU.JO","VOD.JO","WHL.JO","ANG.JO","EXX.JO","GFI.JO",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # BELGIUM — BEL 20
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Belgium",
        "country_code": "BE",
        "index_name": "BEL20",
        "url": "https://en.wikipedia.org/wiki/BEL20",
        "cols": ["Ticker", "Symbol"],
        "manual": [
            # Verified Euronext Brussels (.BR) — April 2026
            # ELI.BR = Elia Group (ELIA.BR was wrong)
            # MELE.BR = Melexis (MELX.BR was wrong)
            # Duplicate UCB.BR removed
            "UCB.BR","KBC.BR","ACKB.BR","ABI.BR","SOLB.BR","COFB.BR","COLR.BR",
            "ARGX.BR","WDP.BR","BPOST.BR","ELI.BR","GBLB.BR","ING.BR","LOTB.BR",
            "MELE.BR","ONTEX.BR","UMI.BR","XIOR.BR",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # PORTUGAL — PSI 20
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Portugal",
        "country_code": "PT",
        "index_name": "PSI20",
        "url": "https://en.wikipedia.org/wiki/PSI-20",
        "cols": ["Ticker", "Symbol"],
        # Wikipedia re-injects these on every run despite them being 404 on yfinance.
        # The exclude list blocks them at the scraper level permanently.
        "exclude": [
            "ELEC.LS","ENGI.LS","IPMA.LS","MOTA.LS",
            "RAMB.LS","SONC.LS","GVOLT.LS","NFGS.LS",
        ],
        "manual": [
            # Verified Euronext Lisbon (.LS) — April 2026
            # Removed (consistently 404 across all runs):
            #   ELEC.LS, ENGI.LS, IPMA.LS, MOTA.LS, RAMB.LS, SONC.LS, GVOLT.LS, NFGS.LS
            "EDP.LS","GALP.LS","BCP.LS","NOS.LS","JMT.LS","SON.LS","EGL.LS",
            "EDPR.LS","CTT.LS","NVG.LS","ALTR.LS","COR.LS","PHR.LS","RENE.LS","TDSA.LS",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # AUSTRIA — ATX 20
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Austria",
        "country_code": "AT",
        "index_name": "ATX20",
        "url": "https://en.wikipedia.org/wiki/Austrian_Traded_Index",
        "cols": ["Ticker", "Symbol"],
        "manual": [
            "EBS.VI","OMV.VI","VIG.VI","RBI.VI","POST.VI","TKA.VI","EVN.VI",
            "SBO.VI","AT1.VI","ATS.VI","BG.VI","CAI.VI","DO.VI","EAG.VI",
            "FMT.VI","GRB.VI","IIA.VI","KAP.VI","LNZ.VI","MIBA.VI","MOS.VI",
            "NWT.VI","OMV.VI","PAL.VI","PYT.VI","QPS.VI","S.VI","SPI.VI",
            "TKH.VI","TQQ.VI","UNI.VI","VAS.VI","VER.VI","WIE.VI","ZAG.VI",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # IRELAND — ISEQ 20
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Ireland",
        "country_code": "IE",
        "index_name": "ISEQ20",
        "url": "https://en.wikipedia.org/wiki/ISEQ_20",
        "cols": ["Ticker", "Symbol"],
        "manual": [
            # Irish companies verified on yfinance — primarily London (.L) listings
            # CRH moved primary listing to NYSE 2023; Flutter moved to NYSE 2024
            "CRH","FLTR",                           # NYSE primary listings
            "CRH.L","FLTR.L","SKG.L","RYA.L",      # also available on LSE
            "EXPN.L","DCC.L","ICG.L","G4M.L",
            # Euronext Dublin (.IR) — subset that works on yfinance
            "BIRG.IR","AIB.IR","BOI.IR","PTSB.IR",
            "FBD.IR","INM.IR","RDSA.IR","IRM.IR",
            # Additional Irish large caps on LSE
            "SMDS.L","SMURFIT.L","MDGH.L","NWL.L",
            "AERCAP.L","GYM.L","CPL.L","DHG.L",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # MEXICO — IPC / BMV (correct suffixes — use MX not .MX for BMV)
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Mexico",
        "country_code": "MX",
        "index_name": "IPC35",
        "url": "https://en.wikipedia.org/wiki/Indice_de_Precios_y_Cotizaciones",
        "cols": ["Ticker", "Symbol"],
        "manual": [
            # yfinance: Mexican stocks use ticker.MX format
            "AMXL.MX","FEMSAUBD.MX","WALMEX.MX","GMEXICOB.MX","KOFUBL.MX",
            "GFNORTEO.MX","GFBANORTEO.MX","BOLSAA.MX","CEMEXCPO.MX","BIMBOA.MX",
            "GCARSOA1.MX","TLEVICPO.MX","MEGACPO.MX","ALFAA.MX","ASURB.MX",
            "GAPB.MX","OMAB.MX","GENTERA.MX","RA.MX","PINFRA.MX","ORBIA.MX",
            "GRUMAB.MX","KIMBER.MX","LIVEPOLC-1.MX","LAB.MX","LABB.MX",
            "VOLAR.MX","AC.MX","AGUA.MX","ALSEA.MX","AMX.MX","BACHOCOB.MX",
            "CUERVO.MX","FINN13.MX","GICSAB.MX","GPROFUT.MX","GCC.MX",
            "ICHB.MX","LABB.MX","LIVEPOLC1.MX","MFRISCOA-1.MX","PASA.MX",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # NEW ZEALAND — NZX 50
    # ══════════════════════════════════════════════════════════════
    {
        "country": "New Zealand",
        "country_code": "NZ",
        "index_name": "NZX50",
        "url": "https://en.wikipedia.org/wiki/S%26P/NZX_50_Index",
        "cols": ["Ticker", "Symbol"],
        "manual": [
            "FPH.NZ","MEL.NZ","CEN.NZ","ATM.NZ","EBO.NZ","MCY.NZ","PCT.NZ",
            "SUM.NZ","AIA.NZ","HGH.NZ","AIR.NZ","ANZ.NZ","ARG.NZ","CAV.NZ",
            "CDI.NZ","CMO.NZ","CNU.NZ","CVT.NZ","DGL.NZ","DIL.NZ","EBO.NZ",
            "EMF.NZ","EVO.NZ","FCG.NZ","FRE.NZ","FSF.NZ","GNE.NZ","GTK.NZ",
            "HLG.NZ","IFT.NZ","KMD.NZ","MFT.NZ","MHJ.NZ","MNW.NZ","MYR.NZ",
            "NPX.NZ","NZR.NZ","OCA.NZ","PFI.NZ","PGW.NZ","POT.NZ","PPH.NZ",
            "RBD.NZ","RYM.NZ","SCL.NZ","SKC.NZ","SKT.NZ","SML.NZ","SPK.NZ",
            "STU.NZ","THL.NZ","THL.NZ","TPW.NZ","TRA.NZ","TWR.NZ","VCT.NZ",
            "VGL.NZ","VHP.NZ","WBC.NZ","WHS.NZ",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # CHINA — CSI 300 / MSCI China  (GitHub repo: MCHI_MSCI_China_ETF_iShares)
    # Shanghai Stock Exchange (.SS) + Shenzhen Stock Exchange (.SZ)
    # Wikipedia auto-scrape supplements the manual list for Shanghai codes.
    # Shenzhen codes (000xxx.SZ / 300xxx.SZ) are manual — Wikipedia only lists .SS
    # ══════════════════════════════════════════════════════════════
    {
        "country": "China",
        "country_code": "CN",
        "index_name": "CSI300",
        "url": "https://en.wikipedia.org/wiki/CSI_300_Index",
        "cols": ["Code", "Symbol", "Ticker"],
        "manual": [
            # ── Shanghai Stock Exchange (.SS) ─────────────────────
            # CSI 300 / MSCI China largest caps by market cap
            "600519.SS","601318.SS","600036.SS","601398.SS","601857.SS","601939.SS",
            "601288.SS","600030.SS","600276.SS","600887.SS","601166.SS","600900.SS",
            "601628.SS","601601.SS","601088.SS","601390.SS","600050.SS","601668.SS",
            "601800.SS","601899.SS","600009.SS","600104.SS","600048.SS","600309.SS",
            "601919.SS","601186.SS","601328.SS","601111.SS","603259.SS","688599.SS",
            "600690.SS","601138.SS","600298.SS","601012.SS","600745.SS","600703.SS",
            "603501.SS","600588.SS","688981.SS","600406.SS","601766.SS","600031.SS",
            "600941.SS","601360.SS","601688.SS","601658.SS","600011.SS","601669.SS",
            # ── Shenzhen Stock Exchange (.SZ) ─────────────────────
            # These are NOT scraped by Wikipedia (CSI300 page only lists .SS)
            # Included manually: CATL, BYD, Ping An Bank, Luxshare, Meituan,etc.
            "000858.SZ","000333.SZ","002594.SZ","300750.SZ","000001.SZ","002415.SZ",
            "000002.SZ","300015.SZ","002352.SZ","000651.SZ","300999.SZ","002714.SZ",
            "000568.SZ","001979.SZ","002475.SZ","000725.SZ","300760.SZ","002230.SZ",
            "300059.SZ","002304.SZ","000776.SZ","002241.SZ","300014.SZ","002311.SZ",
            "000100.SZ","000538.SZ","000625.SZ","000661.SZ","000703.SZ",
            "002027.SZ","002049.SZ","002129.SZ","002142.SZ","002179.SZ","002236.SZ",
            "002271.SZ","002340.SZ","002371.SZ","002385.SZ","002508.SZ","002555.SZ",
            "002607.SZ","002624.SZ","002648.SZ","002938.SZ","300122.SZ","300124.SZ",
            "300308.SZ","300347.SZ","300408.SZ","300413.SZ","300450.SZ","300661.SZ",
            "300676.SZ","300688.SZ","300759.SZ","300763.SZ","300919.SZ","300979.SZ",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # TAIWAN — TWSE 50 + MSCI Taiwan  (GitHub repo: EWT_Taiwan_MSCI_Ishares)
    # Exchange: Taiwan Stock Exchange (.TW)
    # 4-digit numeric codes — Wikipedia scraper handles these correctly
    # via the numeric-code allowance (not HK, so no zero-padding needed)
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Taiwan",
        "country_code": "TW",
        "index_name": "TWSE50",
        "url": "https://en.wikipedia.org/wiki/TWSE_Taiwan_50",
        "cols": ["Ticker", "Code", "Symbol"],
        "manual": [
            # ── TWSE 50 (Taiwan Blue-Chip Index) ─────────────────
            # #1 by cap: TSMC (2330), followed by MediaTek, Hon Hai, etc.
            "2330.TW","2317.TW","2454.TW","2882.TW","2881.TW","2891.TW","2886.TW",
            "1303.TW","1301.TW","2308.TW","2412.TW","3711.TW","2357.TW","2382.TW",
            "2303.TW","2379.TW","1216.TW","2002.TW","2207.TW","3045.TW","2395.TW",
            "2884.TW","2885.TW","2887.TW","2890.TW","2892.TW","5871.TW","2880.TW",
            "3034.TW","2325.TW","2408.TW","2301.TW","2353.TW","2376.TW","2371.TW",
            # ── MSCI Taiwan extra large-caps ─────────────────────
            "2327.TW","3008.TW","2474.TW","2344.TW","4904.TW","2498.TW","2610.TW",
            "2615.TW","2618.TW","2633.TW","3231.TW","3702.TW","4938.TW","5876.TW",
            "6669.TW","6770.TW","8046.TW","9910.TW","1101.TW","1326.TW","1402.TW",
            "2337.TW","2356.TW","2360.TW","2383.TW","2385.TW","2392.TW","2401.TW",
            "2409.TW","2449.TW","2450.TW","2457.TW","2461.TW","2481.TW",
            # Removed: 2448.TW (consistently no timezone across all runs)
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # TURKEY — BIST 30 + BIST 100  (GitHub repo: MSCI_Turkey)
    # Exchange: Borsa Istanbul (.IS)
    # yfinance suffix: .IS  (registered above as "TR": ".IS")
    # Turkish stocks are weighted by free-float market cap in BIST indices
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Turkey",
        "country_code": "TR",
        "index_name": "BIST30",
        "url": "https://en.wikipedia.org/wiki/BIST_30",
        "cols": ["Ticker", "Symbol", "Code"],
        "manual": [
            # ── BIST 30 core ──────────────────────────────────────
            "GARAN.IS","AKBNK.IS","THYAO.IS","EREGL.IS","KCHOL.IS","SAHOL.IS",
            "SISE.IS","TUPRS.IS","ASELS.IS","BIMAS.IS","TCELL.IS","ISCTR.IS",
            "VAKBN.IS","HALKB.IS","YKBNK.IS","FROTO.IS","TOASO.IS","ENKAI.IS",
            "PETKM.IS","ARCLK.IS","MGROS.IS","PGSUS.IS","CCOLA.IS","EKGYO.IS",
            "ULKER.IS","TAVHL.IS","DOHOL.IS","AGHOL.IS","KRDMD.IS","KOZAA.IS",
            # ── BIST 100 large caps ───────────────────────────────
            "AKSEN.IS","ALARK.IS","ALGYO.IS","BRSAN.IS","CIMSA.IS","DOAS.IS",
            "EGEEN.IS","GESAN.IS","GUBRF.IS","IHLGM.IS","KARSN.IS","KLNMA.IS",
            "KONTR.IS","LOGO.IS","MPARK.IS","NTHOL.IS","ODAS.IS","OTKAR.IS",
            "OYAKC.IS","PARSN.IS","SELEC.IS","SOKM.IS","TATGD.IS","TTKOM.IS",
            "TTRAK.IS","VESTL.IS","ZOREN.IS","SODA.IS","EREGL.IS","AKFGY.IS",
            "AEFES.IS","BFREN.IS","BNTAS.IS","BUCIM.IS","CEMTS.IS","DOHOL.IS",
            "GLYHO.IS","HEKTS.IS","KONYA.IS","LMKDC.IS","MRDIN.IS","NETAS.IS",
            "PRKAB.IS","TSKB.IS","ULAS.IS","VESBE.IS",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # INDONESIA — IDX30 / LQ45  (GitHub repo: MSCI_Indonesia)
    # Exchange: Indonesia Stock Exchange / Bursa Efek Indonesia (.JK)
    # yfinance suffix: .JK  (registered above as "ID": ".JK")
    # LQ45 = 45 most liquid stocks; IDX30 = top 30 by liquidity & cap
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Indonesia",
        "country_code": "ID",
        "index_name": "IDX30",
        "url": "https://en.wikipedia.org/wiki/IDX_Composite",
        "cols": ["Ticker", "Symbol", "Code"],
        "manual": [
            # ── IDX30 / LQ45 core ─────────────────────────────────
            "BBCA.JK","BBRI.JK","TLKM.JK","BMRI.JK","ASII.JK","BBNI.JK",
            "GOTO.JK","ICBP.JK","UNVR.JK","KLBF.JK","GGRM.JK","PGAS.JK",
            "ADRO.JK","INCO.JK","EXCL.JK","SMGR.JK","JSMR.JK","PTBA.JK",
            "ITMG.JK","INDF.JK","MNCN.JK","HMSP.JK","INKP.JK","TKIM.JK",
            "BSDE.JK","PWON.JK","BREN.JK","CPIN.JK","AMMN.JK","ANTM.JK",
            # ── LQ45 extended ────────────────────────────────────
            "MDKA.JK","ESSA.JK","BRPT.JK","MAPI.JK","SMRA.JK","CTRA.JK",
            "BBTN.JK","BDMN.JK","BJBR.JK","BJTM.JK","NISP.JK","BNII.JK",
            "MEGA.JK","PNBN.JK","WSKT.JK","PTPP.JK","WIKA.JK","ADHI.JK",
            "UNTR.JK","AALI.JK","LSIP.JK","SIMP.JK","SMCB.JK","INTP.JK",
            "TBIG.JK","TOWR.JK","MTEL.JK","BUKA.JK","EMTK.JK","DCII.JK",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # SAUDI ARABIA — Tadawul All Share Index (TASI)  (GitHub repo: MSCI_Saudi_Arabia_Ishares)
    # Exchange: Saudi Exchange / Tadawul (.SR)
    # yfinance suffix: .SR  (already in EXCHANGE_SUFFIX as "SA": ".SR")
    # 4-digit numeric codes — largest Arab stock market by cap
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Saudi Arabia",
        "country_code": "SA",
        "index_name": "TASI",
        "url": "https://en.wikipedia.org/wiki/Tadawul_All_Share_Index",
        "cols": ["Ticker", "Symbol", "Code"],
        "manual": [
            # ── TASI blue-chips — 4-digit codes + .SR ─────────────
            # #1: Saudi Aramco (2222), followed by SABIC, Al Rajhi Bank, etc.
            "2222.SR","1120.SR","1180.SR","2350.SR","1010.SR","1030.SR",
            "1211.SR","4190.SR","1050.SR","2010.SR","3030.SR","4030.SR",
            "1020.SR","1060.SR","4300.SR","2040.SR","2060.SR","1080.SR",
            "4050.SR","4240.SR","4170.SR","4160.SR","1150.SR","2330.SR",
            "3020.SR","3060.SR","4280.SR","4001.SR","8010.SR","4200.SR",
            # ── TASI mid-caps ─────────────────────────────────────
            "2280.SR","2020.SR","4130.SR","4140.SR","4150.SR","4180.SR",
            "4220.SR","4260.SR","4270.SR","4290.SR","4310.SR","4320.SR",
            "4330.SR","4340.SR","4345.SR","4347.SR","2290.SR","2300.SR",
            "1214.SR","2380.SR","4002.SR","4004.SR","4020.SR","4061.SR",
            "4110.SR","4210.SR","4230.SR","4250.SR","6010.SR","6012.SR",
            "6020.SR","6040.SR","6050.SR","7010.SR","7020.SR","7030.SR",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # MALAYSIA — FTSE Bursa Malaysia KLCI 30  (GitHub repo: MSCI Malaysia EWM)
    # Exchange: Bursa Malaysia (.KL)
    # yfinance suffix: .KL  (registered above as "MY": ".KL")
    # Malaysian stocks use 4-digit numeric codes (e.g. 1155.KL = Maybank)
    # Weighted by free-float market cap; dominated by banks and commodities
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Malaysia",
        "country_code": "MY",
        "index_name": "KLCI30",
        "url": "https://en.wikipedia.org/wiki/FTSE_Bursa_Malaysia_KLCI",
        "cols": ["Ticker", "Code", "Symbol"],
        "manual": [
            # ── KLCI 30 core — 4-digit numeric codes + .KL ────────
            # Ranked by market cap: Maybank, Public Bank, Tenaga, CIMB first
            "1155.KL","1295.KL","5347.KL","1023.KL","5183.KL",
            "6888.KL","6012.KL","4863.KL","1066.KL","5819.KL",
            "1015.KL","4715.KL","3182.KL","1961.KL","4065.KL",
            "2445.KL","3816.KL","6033.KL","5225.KL","7277.KL",
            "4707.KL","7113.KL","5398.KL","1082.KL","3026.KL",
            "6947.KL","3255.KL","2003.KL","5299.KL","5235.KL",
            # ── Bursa Malaysia large caps beyond KLCI 30 ──────────
            "5246.KL","7230.KL","5168.KL","5285.KL","1597.KL",
            "5199.KL","4588.KL","5238.KL","4197.KL","2291.KL",
            "0166.KL","0008.KL","5020.KL","5071.KL","5264.KL",
            "1724.KL","2828.KL","5184.KL","4197.KL","7084.KL",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # THAILAND — SET50 Index  (GitHub repo: THD MSCI Thailand iShares)
    # Exchange: Stock Exchange of Thailand (.BK)
    # yfinance suffix: .BK  (registered above as "TH": ".BK")
    # Thai stocks use alphabetic tickers (named, not numeric)
    # SET50 weighted by free-float market cap; PTT, banks, and airports dominant
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Thailand",
        "country_code": "TH",
        "index_name": "SET50",
        "url": "https://en.wikipedia.org/wiki/Stock_Exchange_of_Thailand",
        "cols": ["Ticker", "Symbol", "Code"],
        "manual": [
            # ── SET50 core — alphabetic tickers + .BK ─────────────
            # Energy & utilities: PTT, PTTEP, GULF, EA, RATCH dominant
            "PTT.BK","PTTEP.BK","OR.BK","PTTGC.BK","TOP.BK","IRPC.BK",
            "GULF.BK","BGRIM.BK","RATCH.BK","EA.BK","GPSC.BK",
            # Banks & finance
            "KBANK.BK","SCBX.BK","BBL.BK","KTB.BK","TTB.BK","TISCO.BK",
            "MTC.BK","SAWAD.BK","AEONTS.BK","KTC.BK",
            # Transport & airports
            "AOT.BK","BEM.BK","AAV.BK","BA.BK",
            # Telecom & tech
            "ADVANC.BK","TRUE.BK","DTAC.BK","DELTA.BK","KCE.BK",
            # Retail & consumer
            "CPALL.BK","HMPRO.BK","CRC.BK","MAKRO.BK","BJC.BK",
            "CENTEL.BK","MINT.BK","OSP.BK","CBG.BK","TU.BK",
            # Healthcare
            "BDMS.BK","BH.BK","CHG.BK",
            # Industry & materials
            "SCC.BK","IVL.BK","SCGP.BK","WHA.BK","STEC.BK","AMATA.BK",
            # Real estate
            "CPN.BK","LH.BK","AWC.BK","SIRI.BK","QH.BK",
            # Other large caps
            "PLANB.BK","ROBINS.BK","BEAUTY.BK","HANA.BK","KSL.BK",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # ISRAEL — TA-35 Index  (GitHub repo: EIS MSCI Israel iShares)
    # Exchange: Tel Aviv Stock Exchange (.TA)
    # yfinance suffix: .TA  (already in registry as "IL": ".TA")
    # TA-35 = 35 largest companies on TASE by market cap
    # Mix of global tech firms (also on NASDAQ) and domestic banks
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Israel",
        "country_code": "IL",
        "index_name": "TA35",
        "url": "https://en.wikipedia.org/wiki/TA-35_Index",
        "cols": ["Ticker", "Symbol", "Code"],
        "manual": [
            # ── Banks (dominate TA-35 by weight) ──────────────────
            "HAPO.TA","LUMI.TA","DSCT.TA","MZTF.TA","FIBI.TA","MERT.TA",
            # ── Technology ────────────────────────────────────────
            "NICE.TA","CHKP.TA","NVMI.TA","KARO.TA","FORTY.TA",
            # ── Pharma ────────────────────────────────────────────
            "TEVA.TA",
            # ── Defence & Aerospace ───────────────────────────────
            "ESLT.TA","ARBE.TA",
            # ── Chemicals & Industry ──────────────────────────────
            "ICL.TA","DLEKG.TA",
            # ── Telecom ───────────────────────────────────────────
            "BEZQ.TA","ORAN.TA","CEL.TA",
            # ── Insurance & Financial ─────────────────────────────
            "PHOE.TA","HARL.TA","MGDL.TA","ALHE.TA","IBI.TA",
            # ── Energy ────────────────────────────────────────────
            "ENLT.TA","NWMD.TA",
            # ── Real Estate ───────────────────────────────────────
            "AZRG.TA","MIVNE.TA","AMOT.TA","GEV.TA","ELCO.TA",
            # ── Other large caps ──────────────────────────────────
            "SPEN.TA","SPNS.TA","BCOM.TA","WLWH.TA",
        ],
    },

    # ══════════════════════════════════════════════════════════════
    # GREECE — FTSE/ATHEX 20  (GitHub repo: GREK ETF / MSCI Greece)
    # Exchange: Athens Stock Exchange / ATHEX (.AT)
    # yfinance suffix: .AT  (already in registry as "GR": ".AT")
    # FTSE/ATHEX Large Cap = 20 largest by free-float market cap
    # Banks dominate (~40% weight); also energy, telco, cement
    # ══════════════════════════════════════════════════════════════
    {
        "country": "Greece",
        "country_code": "GR",
        "index_name": "ATHEX20",
        "url": "https://en.wikipedia.org/wiki/FTSE/ATHEX_Large_Cap",
        "cols": ["Ticker", "Symbol", "Code"],
        "manual": [
            # ── Banks (~40% index weight) ─────────────────────────
            "ETE.AT","EUROB.AT","ALPHA.AT","TPEIR.AT",
            # ── Telecom ───────────────────────────────────────────
            "HTO.AT",
            # ── Energy & Utilities ────────────────────────────────
            "PPC.AT","MOH.AT","ELPE.AT","ADMIE.AT","GEK.AT",
            # ── Industry & Materials ──────────────────────────────
            "MYTIL.AT","TITK.AT","ELLAKTOR.AT",
            # ── Gaming & Lottery ──────────────────────────────────
            "OPAP.AT",
            # ── Real Estate ───────────────────────────────────────
            "LAMDA.AT",
            # ── Transport ─────────────────────────────────────────
            "AEGN.AT",
            # ── Retail & Consumer ─────────────────────────────────
            "INKAT.AT","MERMEREN.AT","BELA.AT",
            # ── Additional ATHEX large caps ───────────────────────
            "EXAE.AT","GEBKA.AT","KEKR.AT","FOURLIS.AT",
            "CENK.AT","PLAT.AT","QUAL.AT","SARANTIS.AT",
        ],
    },

]

# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# CONTINENT GROUPS  (use --group to run a subset)
# ═══════════════════════════════════════════════════════════════════════════════

GROUPS = {
    "americas": ["USA", "Canada", "Brazil", "Mexico"],
    "europe":   ["Germany", "France", "UK", "Spain", "Netherlands", "Switzerland",
                 "Sweden", "Italy", "Denmark", "Norway", "Finland", "Poland",
                 "Belgium", "Portugal", "Austria", "Ireland", "Greece", "Israel"],
    "asia":     ["Japan", "Australia", "India", "South Korea", "Hong Kong",
                 "Singapore", "New Zealand", "South Africa",
                 "China", "Taiwan", "Turkey", "Indonesia", "Saudi Arabia",
                 "Malaysia", "Thailand"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

def run_all(jobs: list):
    """Run all countries one by one with a short pause between each."""
    clean_double_suffix_from_cache()
    total = len(jobs)
    for idx, job in enumerate(jobs, 1):
        log.info(f"\n[{idx}/{total}] Starting {job['country']}...")
        extract_country_data(job)
        if idx < total:
            log.info(f"   Pausing {INTER_COUNTRY_DELAY}s before next country...")
            time.sleep(INTER_COUNTRY_DELAY)

def run_group(group_name: str):
    """Run a continent group only."""
    group_name = group_name.lower()
    if group_name not in GROUPS:
        log.error(f"Unknown group '{group_name}'. Choose from: {list(GROUPS.keys())}")
        return
    names = GROUPS[group_name]
    jobs = [j for j in COUNTRY_JOBS if j["country"] in names]
    log.info(f"Group '{group_name}': {len(jobs)} countries → {names}")
    run_all(jobs)

def run_single(country_name: str):
    """Run a single named country."""
    clean_double_suffix_from_cache()
    matches = [j for j in COUNTRY_JOBS if j["country"].lower() == country_name.lower()]
    if not matches:
        log.error(f"Country '{country_name}' not found.")
        log.info(f"Available: {[j['country'] for j in COUNTRY_JOBS]}")
        return
    extract_country_data(matches[0])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Global Stock Data Extractor — sequential, continent-aware",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python script.py                         # all 37 countries\n"
            "  python script.py --group europe          # Europe only (18)\n"
            "  python script.py --group americas        # Americas only (4)\n"
            "  python script.py --group asia            # Asia/MiddleEast/Africa (15)\n"
            "  python script.py --country Malaysia      # single country\n"
            "  python script.py --country Greece\n"
            "  python script.py --country 'Saudi Arabia'\n"
        )
    )
    parser.add_argument("--country", type=str, default=None,
                        help="Run for a single country (e.g. --country Brazil)")
    parser.add_argument("--group", type=str, default=None,
                        help="Run a continent group: americas | europe | asia")
    args = parser.parse_args()

    if args.country:
        run_single(args.country)
    elif args.group:
        run_group(args.group)
    else:
        run_all(COUNTRY_JOBS)


# Extract-ETF-Data-50-Year-World-Wide
Extract ETF Data >50 Year World Wide

Main Overview of the Python Script

import pandas as pd
import yfinance as yf
import os

tickers = [
    "^GSPC", "^NDX", "^STOXX", "^GDAXI", "EWG", 
    "MCHI", "EWJ", "INDA", "EWT", "EWY", 
    "EWH", "EWP", "EWZ", "EWW", "PAK"
]

start = "1976-01-01"
end = "2026-02-21"

print("Downloading daily data... This will be a large file.")

# 1. Download data (Daily is the default)
data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]

# 2. SKIP RESAMPLING: We jump straight to the transpose
# This keeps every single trading day as a column
df_wide = data.transpose()

# 3. Clean Date headers to 'YYYY-MM-DD'
# We must include the day (%d) so headers are unique
df_wide.columns = df_wide.columns.strftime('%Y-%m-%d')

# 4. Save to CSV
file_name = "etf_historical_DAILY_WIDE.csv"
df_wide.to_csv(file_name)

print("-" * 30)
print(f"Success! Daily file saved at: {os.getcwd()}/{file_name}")
print(f"Total Columns (Days): {len(df_wide.columns)}")
print("Preview (First 5 tickers, first 5 days):")
print(df_wide.iloc[:5, :5])
