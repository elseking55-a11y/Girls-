import json
import time
import threading
from datetime import datetime
from flask import Flask, jsonify, render_template_string
import websocket
import pandas as pd
import numpy as np

app = Flask(__name__)

# System Configurations
SYMBOL = "frxXAUUSD"
APP_ID = "1089"
GRANULARITY = 900  # 15-Minute Candles

# Shared State Storage
candle_history = []
latest_data = {
    "price": 0.0,
    "last_updated": "Initializing...",
    "signal": {
        "type": "WAIT",
        "reason": "Connecting to Deriv real-time feed...",
        "entry": 0.0,
        "sl": 0.0,
        "tp1": 0.0,
        "tp2": 0.0,
        "tp3": 0.0,
        "time": "--"
    }
}

def calculate_indicators(df):
    """Calculates EMA 9, EMA 21, ATR 14, and RSI 14."""
    # EMA
    df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    ema_gain = gain.ewm(com=13, adjust=False).mean()
    ema_loss = loss.ewm(com=13, adjust=False).mean()
    rs = ema_gain / ema_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def analyze_candles(df):
    """Evaluates market state and generates BUY, SELL, or WAIT with reasons."""
    global latest_data
    if len(df) < 25:
        return

    curr = df.iloc[-1]
    close_price = float(curr['close'])
    atr = float(curr['atr']) if not np.isnan(curr['atr']) else 4.0
    rsi = float(curr['rsi']) if not np.isnan(curr['rsi']) else 50.0
    ema_fast = float(curr['ema_fast'])
    ema_slow = float(curr['ema_slow'])
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # 1. BUY CONDITION: EMA 9 above EMA 21 + Bullish RSI (> 52)
    if ema_fast > ema_slow and rsi >= 52:
        latest_data["signal"] = {
            "type": "BUY",
            "reason": f"Bullish trend confirmed. 9 EMA (${ema_fast:.2f}) is above 21 EMA (${ema_slow:.2f}) with buying momentum (RSI: {rsi:.1f}).",
            "entry": round(close_price, 2),
            "sl": round(close_price - (1.5 * atr), 2),
            "tp1": round(close_price + (1.0 * atr), 2),
            "tp2": round(close_price + (2.0 * atr), 2),
            "tp3": round(close_price + (3.5 * atr), 2),
            "time": now_str
        }

    # 2. SELL CONDITION: EMA 9 below EMA 21 + Bearish RSI (< 48)
    elif ema_fast < ema_slow and rsi <= 48:
        latest_data["signal"] = {
            "type": "SELL",
            "reason": f"Bearish trend confirmed. 9 EMA (${ema_fast:.2f}) is below 21 EMA (${ema_slow:.2f}) with selling pressure (RSI: {rsi:.1f}).",
            "entry": round(close_price, 2),
            "sl": round(close_price + (1.5 * atr), 2),
            "tp1": round(close_price - (1.0 * atr), 2),
            "tp2": round(close_price - (2.0 * atr), 2),
            "tp3": round(close_price - (3.5 * atr), 2),
            "time": now_str
        }

    # 3. WAIT CONDITION: Range-bound market, RSI in middle zone (48-52), or conflicting indicators
    else:
        latest_data["signal"] = {
            "type": "WAIT",
            "reason": f"Market consolidating in neutral range (RSI: {rsi:.1f}). EMAs overlap or lack volume. Stand aside until clear breakout.",
            "entry": round(close_price, 2),
            "sl": 0.0,
            "tp1": 0.0,
            "tp2": 0.0,
            "tp3": 0.0,
            "time": now_str
        }

def on_message(ws, message):
    global candle_history, latest_data
    data = json.loads(message)
    
    if data.get("msg_type") == "candles":
        candle_history = data.get("candles", [])
        if candle_history:
            latest_data["price"] = float(candle_history[-1]["close"])
            latest_data["last_updated"] = datetime.utcnow().strftime("%H:%M:%S UTC")
            df = pd.DataFrame(candle_history)
            df = calculate_indicators(df)
            analyze_candles(df)

    elif data.get("msg_type") == "ohlc":
        ohlc = data.get("ohlc", {})
        new_candle = {
            "epoch": ohlc.get("open_time"),
            "open": float(ohlc.get("open")),
            "high": float(ohlc.get("high")),
            "low": float(ohlc.get("low")),
            "close": float(ohlc.get("close")),
        }
        latest_data["price"] = new_candle["close"]
        latest_data["last_updated"] = datetime.utcnow().strftime("%H:%M:%S UTC")

        if candle_history and candle_history[-1]["epoch"] == new_candle["epoch"]:
            candle_history[-1] = new_candle
        else:
            candle_history.append(new_candle)
            
        df = pd.DataFrame(candle_history)
        df = calculate_indicators(df)
        analyze_candles(df)

def start_websocket():
    def run():
        while True:
            try:
                ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
                ws = websocket.WebSocketApp(
                    ws_url,
                    on_open=lambda ws: ws.send(json.dumps({
                        "ticks_history": SYMBOL,
                        "adjust_start_time": 1,
                        "count": 100,
                        "end": "latest",
                        "start": 1,
                        "style": "candles",
                        "granularity": GRANULARITY,
                        "subscribe": 1
                    })),
                    on_message=on_message,
                    on_error=lambda ws, err: print(f"WS Error: {err}")
                )
                ws.run_forever()
            except Exception as e:
                print(f"Reconnecting WS... Error: {e}")
            time.sleep(5)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

start_websocket()

# Embedded Live Dashboard UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gold (XAU/USD) Deriv AI Signals</title>
    <style>
        :root { --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #c9d1d9; --buy: #238636; --sell: #da3633; --wait: #d97706; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; display: flex; justify-content: center; }
        .container { max-width: 520px; width: 100%; }
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 16px; }
        .header { display: flex; justify-content: space-between; align-items: center; }
        .price { font-size: 2rem; font-weight: bold; color: #58a6ff; }
        .badge { font-weight: bold; padding: 6px 14px; border-radius: 20px; font-size: 1.1rem; text-transform: uppercase; }
        .badge-BUY { background: var(--buy); color: white; }
        .badge-SELL { background: var(--sell); color: white; }
        .badge-WAIT { background: var(--wait); color: white; }
        .reason-box { background: #0d1117; border-left: 4px solid #58a6ff; padding: 12px; border-radius: 4px; margin-top: 12px; font-size: 0.95rem; line-height: 1.4; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 15px; }
        .stat-box { background: #0d1117; padding: 12px; border-radius: 8px; border: 1px solid var(--border); }
        .stat-label { font-size: 0.8rem; color: #8b949e; margin-bottom: 4px; }
        .stat-value { font-size: 1.1rem; font-weight: 600; }
        .tp1 { color: #3fb950; } .tp2 { color: #2ea043; } .tp3 { color: #238636; } .sl { color: #f85149; }
        .footer { font-size: 0.75rem; color: #8b949e; text-align: center; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="header">
                <div>
                    <h2 style="margin:0;">Gold (XAU/USD)</h2>
                    <span style="font-size: 0.8rem; color: #8b949e;">Deriv Live Market Stream</span>
                </div>
                <div class="price" id="live-price">$0.00</div>
            </div>
        </div>

        <div class="card">
            <div class="header">
                <span class="stat-label">Market Status</span>
                <span id="signal-badge" class="badge badge-WAIT">WAIT</span>
            </div>

            <div class="reason-box" id="signal-reason">
                Analyzing technical indicators...
            </div>
            
            <div class="grid">
                <div class="stat-box">
                    <div class="stat-label">Current Price</div>
                    <div class="stat-value" id="entry-price">$0.00</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Stop Loss (SL)</div>
                    <div class="stat-value sl" id="sl-price">--</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Take Profit 1</div>
                    <div class="stat-value tp1" id="tp1-price">--</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Take Profit 2</div>
                    <div class="stat-value tp2" id="tp2-price">--</div>
                </div>
                <div class="stat-box" style="grid-column: span 2;">
                    <div class="stat-label">Take Profit 3 (Final Target)</div>
                    <div class="stat-value tp3" id="tp3-price">--</div>
                </div>
            </div>
            <div class="footer">Analysis Time: <span id="signal-time">--</span></div>
        </div>

        <div class="footer">
            Updates continuously • Powered by Deriv Public API<br>
            Last Sync: <span id="last-sync">--</span>
        </div>
    </div>

    <script>
        async function fetchSignalData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                
                document.getElementById('live-price').innerText = '$' + data.price.toFixed(2);
                document.getElementById('last-sync').innerText = data.last_updated;
                
                const sig = data.signal;
                const badge = document.getElementById('signal-badge');
                badge.innerText = sig.type;
                badge.className = 'badge badge-' + sig.type;
                
                document.getElementById('signal-reason').innerText = sig.reason;
                document.getElementById('entry-price').innerText = '$' + sig.entry.toFixed(2);
                document.getElementById('signal-time').innerText = sig.time;

                if (sig.type === "WAIT") {
                    document.getElementById('sl-price').innerText = '--';
                    document.getElementById('tp1-price').innerText = '--';
                    document.getElementById('tp2-price').innerText = '--';
                    document.getElementById('tp3-price').innerText = '--';
                } else {
                    document.getElementById('sl-price').innerText = '$' + sig.sl.toFixed(2);
                    document.getElementById('tp1-price').innerText = '$' + sig.tp1.toFixed(2);
                    document.getElementById('tp2-price').innerText = '$' + sig.tp2.toFixed(2);
                    document.getElementById('tp3-price').innerText = '$' + sig.tp3.toFixed(2);
                }
            } catch (err) {
                console.error("Sync error:", err);
            }
        }
        setInterval(fetchSignalData, 3000);
        fetchSignalData();
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
