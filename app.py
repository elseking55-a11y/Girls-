import json
import time
import threading
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template_string
import websocket
import pandas as pd
import numpy as np

app = Flask(__name__)

SYMBOL = "frxXAUUSD"
APP_ID = "1089"
WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"

TIMEFRAMES = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
}
REQ_TO_TF = {index + 1: tf for index, tf in enumerate(TIMEFRAMES)}

candle_history = {tf: [] for tf in TIMEFRAMES}
latest_data = {
    "price": 0.0,
    "last_updated": "Initializing...",
    "connection": "CONNECTING",
    "connection_error": "",
    "symbol": SYMBOL,
    "timeframes": {},
    "signal": {
        "type": "WAIT",
        "score": 0,
        "confidence": 0,
        "reason": "Connecting to Deriv real-time feed...",
        "entry": 0.0,
        "sl": 0.0,
        "tp1": 0.0,
        "tp2": 0.0,
        "tp3": 0.0,
        "time": "--",
    },
}


def calculate_indicators(df):
    df = df.copy()
    for column in ("open", "high", "low", "close"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    return df.dropna(subset=["close"])


def analyze_timeframe(tf, candles):
    if len(candles) < 25:
        return {
            "direction": "WAIT",
            "score": 0,
            "rsi": 50.0,
            "ema_fast": 0.0,
            "ema_slow": 0.0,
            "atr": 0.0,
            "ready": False,
        }

    df = calculate_indicators(pd.DataFrame(candles))
    if len(df) < 25:
        return {
            "direction": "WAIT",
            "score": 0,
            "rsi": 50.0,
            "ema_fast": 0.0,
            "ema_slow": 0.0,
            "atr": 0.0,
            "ready": False,
        }

    curr = df.iloc[-1]
    close = float(curr["close"])
    fast = float(curr["ema_fast"])
    slow = float(curr["ema_slow"])
    rsi = float(curr["rsi"])
    atr = float(curr["atr"]) if pd.notna(curr["atr"]) else 0.0

    score = 0
    if fast > slow:
        score += 1
    elif fast < slow:
        score -= 1

    if rsi >= 52:
        score += 1
    elif rsi <= 48:
        score -= 1

    # A candle closing above/below the fast EMA adds confirmation.
    if close > fast:
        score += 1
    elif close < fast:
        score -= 1

    if score >= 2:
        direction = "BUY"
    elif score <= -2:
        direction = "SELL"
    else:
        direction = "WAIT"

    return {
        "direction": direction,
        "score": score,
        "rsi": round(rsi, 1),
        "ema_fast": round(fast, 2),
        "ema_slow": round(slow, 2),
        "atr": round(atr, 2),
        "ready": True,
    }


def analyze_all_timeframes():
    results = {}
    for tf, candles in candle_history.items():
        results[tf] = analyze_timeframe(tf, candles)

    latest_data["timeframes"] = results

    ready = [tf for tf, result in results.items() if result["ready"]]
    if not ready:
        return

    # Higher timeframes carry more weight than lower timeframes.
    weights = {"M1": 1, "M5": 1, "M15": 2, "M30": 2, "H1": 3}
    weighted_score = sum(
        results[tf]["score"] * weights[tf]
        for tf in ready
    )
    max_weight = sum(weights[tf] * 3 for tf in ready)

    buy_weight = sum(weights[tf] for tf in ready if results[tf]["direction"] == "BUY")
    sell_weight = sum(weights[tf] for tf in ready if results[tf]["direction"] == "SELL")

    # Require meaningful agreement instead of allowing one timeframe to dominate.
    if weighted_score >= 9 and buy_weight >= 5:
        final_type = "BUY"
    elif weighted_score <= -9 and sell_weight >= 5:
        final_type = "SELL"
    else:
        final_type = "WAIT"

    confidence = int(min(100, abs(weighted_score) / max_weight * 100)) if max_weight else 0
    close_price = latest_data["price"]

    atr_values = [results[tf]["atr"] for tf in ready if results[tf]["atr"] > 0]
    atr = float(np.median(atr_values)) if atr_values else max(close_price * 0.001, 0.01)

    if final_type == "BUY":
        reason = (
            f"Multi-timeframe bullish confirmation: {buy_weight} weighted timeframe points "
            f"support BUY. H1/M30/M15 carry higher weight. Score {weighted_score}."
        )
        sl = close_price - 1.5 * atr
        tp1 = close_price + 1.0 * atr
        tp2 = close_price + 2.0 * atr
        tp3 = close_price + 3.5 * atr
    elif final_type == "SELL":
        reason = (
            f"Multi-timeframe bearish confirmation: {sell_weight} weighted timeframe points "
            f"support SELL. H1/M30/M15 carry higher weight. Score {weighted_score}."
        )
        sl = close_price + 1.5 * atr
        tp1 = close_price - 1.0 * atr
        tp2 = close_price - 2.0 * atr
        tp3 = close_price - 3.5 * atr
    else:
        directions = ", ".join(f"{tf}:{results[tf]['direction']}" for tf in TIMEFRAMES if tf in results)
        reason = (
            f"WAIT: timeframes are not sufficiently aligned ({directions}). "
            f"Weighted score {weighted_score}; waiting for stronger confirmation."
        )
        sl = tp1 = tp2 = tp3 = 0.0

    latest_data["signal"] = {
        "type": final_type,
        "score": weighted_score,
        "confidence": confidence,
        "reason": reason,
        "entry": round(close_price, 2),
        "sl": round(sl, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "tp3": round(tp3, 2),
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


def process_candles(tf, candles):
    if not candles:
        return

    normalized = []
    for candle in candles:
        try:
            normalized.append({
                "epoch": int(candle["epoch"]),
                "open": float(candle["open"]),
                "high": float(candle["high"]),
                "low": float(candle["low"]),
                "close": float(candle["close"]),
            })
        except (KeyError, TypeError, ValueError):
            continue

    candle_history[tf] = normalized[-200:]


def process_ohlc(tf, ohlc):
    if not ohlc:
        return

    try:
        candle = {
            "epoch": int(ohlc["open_time"]),
            "open": float(ohlc["open"]),
            "high": float(ohlc["high"]),
            "low": float(ohlc["low"]),
            "close": float(ohlc["close"]),
        }
    except (KeyError, TypeError, ValueError):
        return

    history = candle_history[tf]
    if history and history[-1]["epoch"] == candle["epoch"]:
        history[-1] = candle
    else:
        history.append(candle)
    candle_history[tf] = history[-200:]

    latest_data["price"] = candle["close"]
    latest_data["connection"] = "CONNECTED"
    latest_data["connection_error"] = ""
    latest_data["last_updated"] = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    analyze_all_timeframes()


def on_message(ws, message):
    try:
        data = json.loads(message)
    except (TypeError, json.JSONDecodeError) as exc:
        latest_data["connection_error"] = f"Invalid Deriv response: {exc}"
        return

    if data.get("error"):
        err = data["error"]
        code = err.get("code", "API_ERROR")
        text_error = err.get("message", "Unknown Deriv API error")
        latest_data["connection"] = "ERROR"
        latest_data["connection_error"] = f"{code}: {text_error}"
        print(f"Deriv API error: {code}: {text_error}", flush=True)
        return

    msg_type = data.get("msg_type")
    req_id = data.get("req_id")

    if msg_type == "candles" and req_id in REQ_TO_TF:
        tf = REQ_TO_TF[req_id]
        process_candles(tf, data.get("candles", []))

        if candle_history[tf]:
            latest_data["price"] = candle_history[tf][-1]["close"]
            latest_data["connection"] = "CONNECTED"
            latest_data["connection_error"] = ""
            latest_data["last_updated"] = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
            analyze_all_timeframes()

    elif msg_type == "ohlc":
        tf = REQ_TO_TF.get(req_id)
        if tf:
            process_ohlc(tf, data.get("ohlc", {}))


def subscribe_all(ws):
    for req_id, (tf, granularity) in enumerate(TIMEFRAMES.items(), start=1):
        request = {
            "ticks_history": SYMBOL,
            "adjust_start_time": 1,
            "count": 100,
            "end": "latest",
            "style": "candles",
            "granularity": granularity,
            "subscribe": 1,
            "req_id": req_id,
        }
        ws.send(json.dumps(request))
        time.sleep(0.15)


def start_websocket():
    def run():
        while True:
            ws = None
            try:
                latest_data["connection"] = "CONNECTING"
                latest_data["connection_error"] = ""

                ws = websocket.WebSocketApp(
                    WS_URL,
                    on_open=lambda socket: subscribe_all(socket),
                    on_message=on_message,
                    on_error=lambda socket, err: (
                        latest_data.update({
                            "connection": "ERROR",
                            "connection_error": str(err),
                        }),
                        print(f"Deriv WS error: {err}", flush=True),
                    ),
                    on_close=lambda socket, code, msg: (
                        latest_data.update({"connection": "RECONNECTING"}),
                        print(f"Deriv WS closed: {code} {msg}", flush=True),
                    ),
                )
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                latest_data["connection"] = "ERROR"
                latest_data["connection_error"] = str(exc)
                print(f"Deriv reconnect error: {exc}", flush=True)
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass

            latest_data["connection"] = "RECONNECTING"
            time.sleep(2)

    threading.Thread(target=run, daemon=True, name="deriv-multi-timeframe").start()


start_websocket()


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
