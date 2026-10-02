import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template_string
import websocket
import pandas as pd
import numpy as np

app = Flask(__name__)

APP_ID = "1089"
WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
DEFAULT_SYMBOL = "frxXAUUSD"
TIMEFRAMES = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600}

latest_data = {
    "price": 0.0, "last_updated": "--", "connection": "CONNECTING",
    "connection_error": "", "symbol": DEFAULT_SYMBOL, "timeframes": {},
    "signal": {"type": "WAIT", "score": 0, "confidence": 0,
               "reason": "Connecting to Deriv real-time feed...", "entry": 0.0,
               "sl": 0.0, "tp1": 0.0, "tp2": 0.0, "tp3": 0.0, "time": "--"}
}


def calculate_indicators(df):
    df = df.copy()
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)
    return df.dropna(subset=["close"])


def analyze_timeframe(candles):
    if len(candles) < 25:
        return {"direction":"WAIT","score":0,"rsi":50.0,"ema_fast":0.0,"ema_slow":0.0,"atr":0.0,"ready":False}
    df = calculate_indicators(pd.DataFrame(candles))
    if len(df) < 25:
        return {"direction":"WAIT","score":0,"rsi":50.0,"ema_fast":0.0,"ema_slow":0.0,"atr":0.0,"ready":False}
    row = df.iloc[-1]
    close, fast, slow = float(row["close"]), float(row["ema_fast"]), float(row["ema_slow"])
    rsi, atr = float(row["rsi"]), float(row["atr"]) if pd.notna(row["atr"]) else 0.0
    score = (1 if fast > slow else -1 if fast < slow else 0)
    score += (1 if rsi >= 52 else -1 if rsi <= 48 else 0)
    score += (1 if close > fast else -1 if close < fast else 0)
    direction = "BUY" if score >= 2 else "SELL" if score <= -2 else "WAIT"
    return {"direction":direction,"score":score,"rsi":round(rsi,1),"ema_fast":round(fast,2),
            "ema_slow":round(slow,2),"atr":round(atr,2),"ready":True}


def get_active_gold_symbol():
    ws = websocket.create_connection(WS_URL, timeout=8)
    try:
        ws.send(json.dumps({"active_symbols":"brief","req_id":100}))
        while True:
            data = json.loads(ws.recv())
            if data.get("error"):
                raise RuntimeError(f'{data["error"].get("code","API_ERROR")}: {data["error"].get("message","Deriv error")}')
            if data.get("req_id") == 100 and data.get("msg_type") == "active_symbols":
                candidates = []
                for item in data.get("active_symbols", []):
                    symbol = item.get("underlying_symbol") or item.get("symbol")
                    name = (item.get("underlying_symbol_name") or item.get("display_name") or "").lower()
                    if symbol and ("xau" in symbol.lower() or "gold" in name):
                        candidates.append(symbol)
                return DEFAULT_SYMBOL if DEFAULT_SYMBOL in candidates else (candidates[0] if candidates else None)
    finally:
        ws.close()


def fetch_timeframe(symbol, tf, granularity):
    ws = websocket.create_connection(WS_URL, timeout=8)
    try:
        ws.send(json.dumps({"ticks_history":symbol,"adjust_start_time":1,"count":100,"end":"latest",
                            "style":"candles","granularity":granularity,"req_id":1}))
        deadline = time.time() + 8
        while time.time() < deadline:
            data = json.loads(ws.recv())
            if data.get("error"):
                raise RuntimeError(f'{data["error"].get("code","API_ERROR")}: {data["error"].get("message","Deriv error")}')
            if data.get("req_id") == 1 and data.get("msg_type") == "candles":
                candles = []
                for c in data.get("candles", []):
                    try:
                        candles.append({"epoch":int(c["epoch"]),"open":float(c["open"]),"high":float(c["high"]),
                                        "low":float(c["low"]),"close":float(c["close"])})
                    except (KeyError,TypeError,ValueError):
                        pass
                return candles
        raise RuntimeError(f"{tf}: Deriv candle request timed out")
    finally:
        ws.close()


def build_snapshot():
    symbol = get_active_gold_symbol()
    if not symbol:
        raise RuntimeError("Deriv has no active XAU/Gold symbol available right now.")
    results = {}
    errors = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs = {pool.submit(fetch_timeframe, symbol, tf, gran): tf for tf, gran in TIMEFRAMES.items()}
        for job in as_completed(jobs):
            tf = jobs[job]
            try:
                results[tf] = analyze_timeframe(job.result())
            except Exception as exc:
                results[tf] = {"direction":"WAIT","score":0,"rsi":50.0,"ema_fast":0.0,"ema_slow":0.0,"atr":0.0,"ready":False}
                errors.append(f"{tf}: {exc}")
    ready = [tf for tf in TIMEFRAMES if results.get(tf, {}).get("ready")]
    if not ready:
        raise RuntimeError("; ".join(errors) or "No Deriv timeframe data received.")
    weights = {"M1":1,"M5":1,"M15":2,"M30":2,"H1":3}
    score = sum(results[tf]["score"] * weights[tf] for tf in ready)
    max_score = sum(weights[tf] * 3 for tf in ready)
    buy_weight = sum(weights[tf] for tf in ready if results[tf]["direction"] == "BUY")
    sell_weight = sum(weights[tf] for tf in ready if results[tf]["direction"] == "SELL")
    signal = "BUY" if score >= 9 and buy_weight >= 5 else "SELL" if score <= -9 and sell_weight >= 5 else "WAIT"
    confidence = int(min(100, abs(score) / max_score * 100)) if max_score else 0
    price_candidates = [results[tf].get("_price") for tf in ready]
    # Get the current price from the fastest fresh candle in a separate lightweight request.
    candles = fetch_timeframe(symbol, "M1", 60)
    price = candles[-1]["close"] if candles else 0.0
    atrs = [results[tf]["atr"] for tf in ready if results[tf]["atr"] > 0]
    atr = float(np.median(atrs)) if atrs else max(price * 0.001, 0.01)
    if signal == "BUY":
        sl,tp1,tp2,tp3 = price-1.5*atr,price+atr,price+2*atr,price+3.5*atr
        reason = f"Multi-timeframe BUY confirmation. Weighted score {score}; buy weight {buy_weight}."
    elif signal == "SELL":
        sl,tp1,tp2,tp3 = price+1.5*atr,price-atr,price-2*atr,price-3.5*atr
        reason = f"Multi-timeframe SELL confirmation. Weighted score {score}; sell weight {sell_weight}."
    else:
        sl=tp1=tp2=tp3=0.0
        reason = "WAIT: M1/M5/M15/M30/H1 are not sufficiently aligned for a final signal."
    latest_data.update({
        "price":price,"last_updated":datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        "connection":"CONNECTED" if not errors else "CONNECTED_WITH_WARNINGS","connection_error":"; ".join(errors),
        "symbol":symbol,"timeframes":results,
        "signal":{"type":signal,"score":score,"confidence":confidence,"reason":reason,
                  "entry":round(price,2),"sl":round(sl,2),"tp1":round(tp1,2),"tp2":round(tp2,2),
                  "tp3":round(tp3,2),"time":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
    })
    return latest_data

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ELISY254 Multi-Timeframe Deriv Signals</title>
<style>
:root{--bg:#090d12;--card:#121820;--border:#28313b;--text:#e6edf3;--muted:#8b98a7;--buy:#19b56b;--sell:#e05252;--wait:#d59b32}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,Segoe UI,sans-serif;padding:16px}.container{max-width:720px;margin:auto}.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:16px;margin-bottom:14px}.top{display:flex;justify-content:space-between;gap:12px;align-items:center}.muted{color:var(--muted);font-size:13px}.price{font-size:30px;font-weight:750}.badge{padding:7px 14px;border-radius:8px;font-weight:800}.BUY{background:var(--buy);color:#06140d}.SELL{background:var(--sell);color:#190707}.WAIT{background:var(--wait);color:#1b1204}.reason{margin-top:14px;padding:12px;background:#0b1016;border-radius:9px;line-height:1.45}.tf-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:14px}.tf{background:#0b1016;border:1px solid var(--border);border-radius:10px;padding:10px;text-align:center}.tf strong{display:block;font-size:13px}.dir{font-weight:800;margin:5px 0}.small{font-size:11px;color:var(--muted)}.targets{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:14px}.target{background:#0b1016;border-radius:9px;padding:10px}.target span{display:block;color:var(--muted);font-size:11px}.target b{font-size:14px}.sl b{color:#f07171}.tp b{color:#39c984}.status{margin-top:12px;font-size:12px;color:var(--muted)}@media(max-width:560px){.tf-grid{grid-template-columns:repeat(3,1fr)}.targets{grid-template-columns:repeat(2,1fr)}.price{font-size:24px}}
</style>
</head>
<body>
<div class="container">
<div class="card">
<div class="top"><div><h2 style="margin:0">Gold (XAU/USD)</h2><div class="muted">Live Deriv multi-timeframe analysis</div></div><div class="price" id="price">$0.00</div></div>
</div>
<div class="card">
<div class="top"><span class="muted">FINAL SIGNAL</span><span id="badge" class="badge WAIT">WAIT</span></div>
<div class="reason" id="reason">Connecting to Deriv...</div>
<div class="tf-grid" id="tf-grid"></div>
<div class="targets">
<div class="target"><span>ENTRY</span><b id="entry">$0.00</b></div>
<div class="target sl"><span>STOP LOSS</span><b id="sl">--</b></div>
<div class="target tp"><span>TP1</span><b id="tp1">--</b></div>
<div class="target tp"><span>TP2</span><b id="tp2">--</b></div>
<div class="target tp"><span>TP3</span><b id="tp3">--</b></div>
<div class="target"><span>CONFIDENCE</span><b id="confidence">0%</b></div>
</div>
<div class="status">Connection: <b id="connection">CONNECTING</b> · Last sync: <span id="sync">--</span></div>
</div>
</div>
<script>
const money=v=>Number(v||0).toFixed(2);
async function update(){
 try{
  const r=await fetch('/api/data',{cache:'no-store'}); const d=await r.json();
  document.getElementById('price').textContent='$'+money(d.price);
  document.getElementById('connection').textContent=d.connection+(d.connection_error?' · '+d.connection_error:'');
  document.getElementById('sync').textContent=d.last_updated;
  const s=d.signal||{}; const b=document.getElementById('badge');
  b.textContent=s.type||'WAIT'; b.className='badge '+(s.type||'WAIT');
  document.getElementById('reason').textContent=s.reason||'Analyzing...';
  document.getElementById('entry').textContent='$'+money(s.entry);
  document.getElementById('confidence').textContent=(s.confidence||0)+'%';
  ['sl','tp1','tp2','tp3'].forEach(k=>document.getElementById(k).textContent=s.type==='WAIT'?'--':'$'+money(s[k]));
  const grid=document.getElementById('tf-grid'); grid.innerHTML='';
  ['M1','M5','M15','M30','H1'].forEach(tf=>{
   const x=(d.timeframes||{})[tf]||{};
   grid.innerHTML+=`<div class="tf"><strong>${tf}</strong><div class="dir ${x.direction||'WAIT'}">${x.direction||'WAIT'}</div><div class="small">RSI ${x.rsi??'--'}</div><div class="small">Score ${x.score??0}</div></div>`;
  });
 }catch(e){document.getElementById('connection').textContent='RECONNECTING';}
}
update(); setInterval(update,2000);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/data")
def get_data():
    return jsonify(latest_data)


@app.route("/health")
def health():
    return jsonify({
        "ok": latest_data["connection"] == "CONNECTED",
        "connection": latest_data["connection"],
        "symbol": latest_data["symbol"],
        "timeframes_ready": [
            tf for tf, data in latest_data["timeframes"].items()
            if data.get("ready")
        ],
        "last_updated": latest_data["last_updated"],
        "error": latest_data["connection_error"],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
