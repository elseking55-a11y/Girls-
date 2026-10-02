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
WS_URL = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"
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


def fetch_market_snapshot():
    ws = websocket.create_connection(WS_URL, timeout=10)
    try:
        ws.send(json.dumps({"active_symbols":"brief","product_type":"basic","req_id":100}))
        symbol = None
        candles = {}
        deadline = time.time() + 10
        while time.time() < deadline and (symbol is None or len(candles) < len(TIMEFRAMES)):
            data = json.loads(ws.recv())
            if data.get("error"):
                err=data["error"]
                raise RuntimeError(f'{err.get("code","API_ERROR")}: {err.get("message","Deriv error")}')
            if data.get("msg_type") == "active_symbols" and data.get("req_id") == 100:
                candidates=[]
                for item in data.get("active_symbols",[]):
                    sym=item.get("underlying_symbol") or item.get("symbol")
                    name=(item.get("underlying_symbol_name") or item.get("display_name") or "").lower()
                    if sym and ("xau" in sym.lower() or "gold" in name): candidates.append(sym)
                symbol=DEFAULT_SYMBOL if DEFAULT_SYMBOL in candidates else (candidates[0] if candidates else None)
                if not symbol: raise RuntimeError("Deriv has no active XAU/Gold symbol available right now.")
                for req_id,(tf,gran) in enumerate(TIMEFRAMES.items(),1):
                    ws.send(json.dumps({"ticks_history":symbol,"adjust_start_time":1,"count":100,"end":"latest","style":"candles","granularity":gran,"req_id":req_id}))
            req_id=data.get("req_id")
            if data.get("msg_type")=="candles" and isinstance(req_id,int) and 1<=req_id<=5:
                tf=list(TIMEFRAMES)[req_id-1]
                parsed=[]
                for x in data.get("candles",[]):
                    try: parsed.append({"epoch":int(x["epoch"]),"open":float(x["open"]),"high":float(x["high"]),"low":float(x["low"]),"close":float(x["close"])})
                    except (KeyError,TypeError,ValueError): pass
                candles[tf]=parsed
        if symbol is None: raise RuntimeError("Deriv did not return an active Gold/XAU symbol.")
        missing=[tf for tf in TIMEFRAMES if tf not in candles]
        if missing: raise RuntimeError("Deriv data timeout for: "+", ".join(missing))
        return symbol,candles
    finally:
        ws.close()


def build_snapshot():
    symbol,candle_sets=fetch_market_snapshot()
    results={tf:analyze_timeframe(candle_sets[tf]) for tf in TIMEFRAMES}
    ready=[tf for tf in TIMEFRAMES if results[tf]["ready"]]
    weights={"M1":1,"M5":1,"M15":2,"M30":2,"H1":3}
    score=sum(results[tf]["score"]*weights[tf] for tf in ready)
    max_score=sum(weights[tf]*3 for tf in ready)
    buy_weight=sum(weights[tf] for tf in ready if results[tf]["direction"]=="BUY")
    sell_weight=sum(weights[tf] for tf in ready if results[tf]["direction"]=="SELL")
    signal="BUY" if score>=9 and buy_weight>=5 else "SELL" if score<=-9 and sell_weight>=5 else "WAIT"
    confidence=int(min(100,abs(score)/max_score*100)) if max_score else 0
    price=candle_sets["M1"][-1]["close"] if candle_sets["M1"] else 0.0
    atrs=[results[tf]["atr"] for tf in ready if results[tf]["atr"]>0]
    atr=float(np.median(atrs)) if atrs else max(price*0.001,0.01)
    if signal=="BUY": sl,tp1,tp2,tp3=price-1.5*atr,price+atr,price+2*atr,price+3.5*atr
    elif signal=="SELL": sl,tp1,tp2,tp3=price+1.5*atr,price-atr,price-2*atr,price-3.5*atr
    else: sl=tp1=tp2=tp3=0.0
    reason=(f"Multi-timeframe {signal} confirmation. Weighted score {score}." if signal!="WAIT" else "WAIT: M1/M5/M15/M30/H1 are not sufficiently aligned for a final signal.")
    latest_data.update({"price":price,"last_updated":datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),"connection":"CONNECTED","connection_error":"","symbol":symbol,"timeframes":results,"signal":{"type":signal,"score":score,"confidence":confidence,"reason":reason,"entry":round(price,2),"sl":round(sl,2),"tp1":round(tp1,2),"tp2":round(tp2,2),"tp3":round(tp3,2),"time":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}})
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
const TF={M1:60,M5:300,M15:900,M30:1800,H1:3600};
let ws,retry,req={},candles={};
function status(x){document.getElementById('connection').textContent=x}
function calc(a){
 if(a.length<25)return {direction:'WAIT',score:0,rsi:50,atr:0,ready:false};
 let c=a.map(x=>x.close),h=a.map(x=>x.high),l=a.map(x=>x.low),e9=c[0],e21=c[0];
 for(let i=1;i<c.length;i++){e9=c[i]*.2+e9*.8;e21=c[i]*(2/22)+e21*(20/22)}
 let g=0,lo=0;for(let i=Math.max(1,c.length-14);i<c.length;i++){let d=c[i]-c[i-1];d>=0?g+=d:lo-=d}
 let r=lo?100-100/(1+g/lo):100,tr=[];for(let i=1;i<c.length;i++)tr.push(Math.max(h[i]-l[i],Math.abs(h[i]-c[i-1]),Math.abs(l[i]-c[i-1])));
 let atr=tr.slice(-14).reduce((a,b)=>a+b,0)/Math.min(14,tr.length);
 let s=(e9>e21?1:-1)+(r>=52?1:r<=48?-1:0)+(c.at(-1)>e9?1:-1);
 return {direction:s>=2?'BUY':s<=-2?'SELL':'WAIT',score:s,rsi:+r.toFixed(1),atr:+atr.toFixed(2),ready:true};
}
function render(){
 let o={},w={M1:1,M5:1,M15:2,M30:2,H1:3},score=0,buy=0,sell=0,max=0,n=0;
 for(let tf in TF){o[tf]=calc(candles[tf]||[]);let x=o[tf];if(x.ready){n++;score+=x.score*w[tf];max+=3*w[tf];if(x.direction==='BUY')buy+=w[tf];if(x.direction==='SELL')sell+=w[tf]}}
 let type=score>=9&&buy>=5?'BUY':score<=-9&&sell>=5?'SELL':'WAIT',p=candles.M1?.at(-1)?.close||0;
 let aa=Object.values(o).map(x=>x.atr).filter(x=>x>0).sort((a,b)=>a-b),a=aa.length?aa[Math.floor(aa.length/2)]:p*.001;
 let sl=type==='BUY'?p-1.5*a:type==='SELL'?p+1.5*a:0;
 let t1=type==='BUY'?p+a:type==='SELL'?p-a:0,t2=type==='BUY'?p+2*a:type==='SELL'?p-2*a:0,t3=type==='BUY'?p+3.5*a:type==='SELL'?p-3.5*a:0;
 document.getElementById('price').textContent='$'+money(p);document.getElementById('entry').textContent='$'+money(p);
 let b=document.getElementById('badge');b.textContent=type;b.className='badge '+type;
 document.getElementById('reason').textContent=n===5?'Multi-timeframe '+type+' analysis. Weighted score '+score+'.':'Receiving live M1/M5/M15/M30/H1 candles...';
 document.getElementById('confidence').textContent+(max?Math.round(Math.abs(score)/max*100):0)+'%';
 for(let [id,v] of Object.entries({sl,tp1:t1,tp2:t2,tp3:t3}))document.getElementById(id).textContent=type==='WAIT'?'--':'$'+money(v);
 document.getElementById('tf-grid').innerHTML=Object.keys(TF).map(tf=>{let x=o[tf];return '<div class="tf"><strong>'+tf+'</strong><div class="dir '+x.direction+'">'+x.direction+'</div><div class="small">RSI '+(x.ready?x.rsi:'--')+'</div><div class="small">Score '+x.score+'</div></div>'}).join('');
 document.getElementById('sync').textContent=new Date().toISOString().replace('T',' ').replace('Z',' UTC');
}
function connect(){
 clearTimeout(retry);try{if(ws)ws.close()}catch(e){} candles={};req={};status('CONNECTING TO DERIV LIVE DATA');
 ws=new WebSocket('wss://ws.binaryws.com/websockets/v3?app_id=1089');
 ws.onopen=()=>{status('CONNECTED · REQUESTING MARKET DATA');ws.send(JSON.stringify({active_symbols:'brief',product_type:'basic',req_id:100}))};
 ws.onmessage=e=>{let d=JSON.parse(e.data);if(d.error){status('DERIV ERROR · '+d.error.message);return}
  if(d.msg_type==='active_symbols'&&d.req_id===100){let a=d.active_symbols||[],g=a.find(x=>x.symbol==='frxXAUUSD')||a.find(x=>/xau|gold/i.test((x.symbol||'')+' '+(x.display_name||'')));if(!g){status('DERIV ERROR · XAU/USD unavailable');return}
   Object.entries(TF).forEach(([tf,gran],i)=>{req[i+1]=tf;ws.send(JSON.stringify({ticks_history:g.symbol,adjust_start_time:1,count:100,end:'latest',style:'candles',granularity:gran,req_id:i+1}))});status('CONNECTED · LOADING M1 M5 M15 M30 H1')}
  if(d.msg_type==='candles'&&req[d.req_id]){candles[req[d.req_id]]=(d.candles||[]).map(x=>({epoch:+x.epoch,open:+x.open,high:+x.high,low:+x.low,close:+x.close}));render();if(Object.keys(candles).length===5)status('CONNECTED · LIVE DERIV DATA · XAU/USD')}
 };
 ws.onerror=()=>status('DERIV DISCONNECTED · RECONNECTING');ws.onclose=()=>{status('DERIV DISCONNECTED · RECONNECTING');retry=setTimeout(connect,1500)};
}
connect();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/data")
def get_data():
    try:
        snapshot = build_snapshot()
        return jsonify(snapshot)
    except Exception as exc:
        latest_data["connection"] = "ERROR"
        latest_data["connection_error"] = str(exc)
        latest_data["last_updated"] = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        return jsonify(latest_data), 200


@app.route("/health")
def health():
    return jsonify({
        "ok": latest_data["connection"] in ("CONNECTED", "CONNECTED_WITH_WARNINGS"),
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
