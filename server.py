"""
server.py  —  Flask log szerver TFLite detekciókhoz
Futtatás: python server.py
Elérhető: http://<PC_IP>:5000  (ugyanazon WiFi-n telefonról is)
"""

from flask import Flask, request, jsonify, Response
from datetime import datetime
import json, queue, threading

app = Flask(__name__)

# In-memory log tároló (max 500 bejegyzés)
logs = []
MAX_LOGS = 500
logs_lock = threading.Lock()

# SSE (Server-Sent Events) queue – minden csatlakozott kliensnek
sse_clients = []
sse_lock = threading.Lock()

# ─────────────────────────────────────────────
# C program ide POST-olja a detekciós eredményt
# ─────────────────────────────────────────────
@app.route("/log", methods=["POST"])
def receive_log():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "invalid JSON"}), 400

    entry = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label":      data.get("label", "unknown"),
        "confidence": round(float(data.get("confidence", 0.0)), 4),
        "position":   data.get("position", {"x": 0, "y": 0}),
        "image_b64":  data.get("image_b64", ""),
    }

    with logs_lock:
        logs.append(entry)
        if len(logs) > MAX_LOGS:
            logs.pop(0)

    # SSE broadcast minden kliensnek
    event_data = f"data: {json.dumps(entry)}\n\n"
    with sse_lock:
        dead = []
        for q in sse_clients:
            try:
                q.put_nowait(event_data)
            except Exception:
                dead.append(q)
        for q in dead:
            sse_clients.remove(q)

    return jsonify({"status": "ok"}), 200


# ─────────────────────────────────────────────
# SSE stream – böngésző ide figyel
# ─────────────────────────────────────────────
@app.route("/stream")
def sse_stream():
    q = queue.Queue(maxsize=50)
    with sse_lock:
        sse_clients.append(q)

    def generate():
        try:
            while True:
                data = q.get(timeout=30)
                yield data
        except Exception:
            pass
        finally:
            with sse_lock:
                if q in sse_clients:
                    sse_clients.remove(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ─────────────────────────────────────────────
# Eddigi logok lekérdezése (oldalbetöltéskor)
# ─────────────────────────────────────────────
@app.route("/logs")
def get_logs():
    with logs_lock:
        return jsonify(list(reversed(logs)))   # legújabb elöl


# ─────────────────────────────────────────────
# HTML dashboard (telefon + PC böngésző)
# ─────────────────────────────────────────────
@app.route("/")
def dashboard():
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="hu">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Detekciós Log</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  :root {
    --bg:      #0d0f14;
    --surface: #161921;
    --border:  #252a36;
    --accent:  #00e5a0;
    --accent2: #6b7fff;
    --warn:    #ffb547;
    --danger:  #ff5c5c;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --mono:    'IBM Plex Mono', monospace;
    --sans:    'IBM Plex Sans', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; background: var(--bg); color: var(--text); font-family: var(--sans); }

  header {
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0; background: var(--bg); z-index: 10;
  }
  header h1 { font-size: 15px; font-weight: 600; letter-spacing: .05em; color: var(--accent); font-family: var(--mono); }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); transition: background .3s; }
  .status-dot.live { background: var(--accent); box-shadow: 0 0 8px var(--accent); animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  .stats {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px;
    background: var(--border); border-bottom: 1px solid var(--border);
  }
  .stat { background: var(--surface); padding: 12px 16px; text-align: center; }
  .stat-val { font-family: var(--mono); font-size: 22px; font-weight: 600; color: var(--accent); }
  .stat-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; margin-top: 2px; }

  .filter-bar {
    padding: 10px 16px; display: flex; gap: 8px; flex-wrap: wrap;
    border-bottom: 1px solid var(--border); background: var(--surface);
  }
  .filter-bar input {
    flex: 1; min-width: 120px; background: var(--bg); border: 1px solid var(--border);
    color: var(--text); padding: 6px 10px; border-radius: 4px; font-family: var(--mono); font-size: 12px;
  }
  .filter-bar input:focus { outline: none; border-color: var(--accent); }
  .btn {
    padding: 6px 12px; border-radius: 4px; border: 1px solid var(--border);
    background: var(--bg); color: var(--muted); font-size: 11px; cursor: pointer;
    font-family: var(--mono); transition: all .2s;
  }
  .btn:hover { border-color: var(--accent); color: var(--accent); }
  .btn.danger:hover { border-color: var(--danger); color: var(--danger); }

  #log-container { padding: 12px 16px; display: flex; flex-direction: column; gap: 6px; }

  .log-entry {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 4px; padding: 10px 12px;
    font-family: var(--mono); font-size: 12px;
    animation: slideIn .25s ease;
  }
  .log-row {
    display: grid; grid-template-columns: auto 1fr auto auto;
    gap: 8px; align-items: center;
  }
  .log-pos { font-size: 10px; color: var(--muted); white-space: nowrap; }
  .log-img {
    margin-top: 8px; width: 100%; max-width: 240px;
    border-radius: 4px; border: 1px solid var(--border);
    display: block;
  }
  .log-entry.high { border-left-color: var(--accent); }
  .log-entry.low  { border-left-color: var(--muted);  opacity: .7; }
  .log-entry.warn { border-left-color: var(--warn);   }

  @keyframes slideIn { from { opacity:0; transform: translateY(-6px); } to { opacity:1; transform: none; } }

  .log-ts    { color: var(--muted); font-size: 10px; white-space: nowrap; }
  .log-label { color: var(--text); font-weight: 600; font-size: 13px; }
  .log-conf  { 
    font-size: 11px; padding: 2px 7px; border-radius: 3px;
    background: rgba(0,229,160,.1); color: var(--accent);
    white-space: nowrap;
  }
  .log-conf.med  { background: rgba(107,127,255,.1); color: var(--accent2); }
  .log-conf.low  { background: rgba(100,116,139,.1); color: var(--muted);   }

  .empty { text-align: center; padding: 60px 20px; color: var(--muted); font-family: var(--mono); font-size: 13px; }

  @media (max-width: 480px) {
    .stats { grid-template-columns: repeat(3,1fr); }
    .stat-val { font-size: 18px; }
    .log-entry { grid-template-columns: 1fr auto; grid-template-rows: auto auto; }
    .log-ts { grid-column: 1/-1; font-size: 9px; }
  }
</style>
</head>
<body>

<header>
  <h1>▶ DETEKCIÓS LOG</h1>
  <div style="display:flex;align-items:center;gap:8px">
    <span id="status-text" style="font-size:11px;color:var(--muted);font-family:var(--mono)">Kapcsolódás…</span>
    <div class="status-dot" id="dot"></div>
  </div>
</header>

<div class="stats">
  <div class="stat">
    <div class="stat-val" id="stat-total">0</div>
    <div class="stat-label">Összes</div>
  </div>
  <div class="stat">
    <div class="stat-val" id="stat-session">0</div>
    <div class="stat-label">Munkamenet</div>
  </div>
  <div class="stat">
    <div class="stat-val" id="stat-conf">—</div>
    <div class="stat-label">Átl. konfidencia</div>
  </div>
</div>

<div class="filter-bar">
  <input type="text" id="filter-input" placeholder="Szűrés label alapján…" oninput="filterLogs()">
  <button class="btn" onclick="togglePause()" id="pause-btn">⏸ Pause</button>
  <button class="btn danger" onclick="clearLogs()">✕ Törlés</button>
</div>

<div id="log-container">
  <div class="empty" id="empty-msg">Várakozás a C program adataira…</div>
</div>

<script>
let allLogs = [], sessionCount = 0, paused = false, filterTerm = '';

function confClass(c) {
  if (c >= 0.75) return 'high';
  if (c >= 0.45) return '';
  return 'low';
}
function confBadgeClass(c) {
  if (c >= 0.75) return '';
  if (c >= 0.45) return 'med';
  return 'low';
}

function renderEntry(entry, prepend=true) {
  if (filterTerm && !entry.label.toLowerCase().includes(filterTerm)) return;
  document.getElementById('empty-msg')?.remove();

  const div = document.createElement('div');
  div.className = `log-entry ${confClass(entry.confidence)}`;
  const pos = entry.position ? `(${entry.position.x}, ${entry.position.y})` : '';
  div.innerHTML = `
    <div class="log-row">
      <span class="log-ts">${entry.timestamp}</span>
      <span class="log-label">${entry.label}</span>
      <span class="log-conf ${confBadgeClass(entry.confidence)}">${(entry.confidence*100).toFixed(1)}%</span>
      <span class="log-pos">${pos}</span>
    </div>
    ${entry.image_b64 ? `<img class="log-img" src="data:image/jpeg;base64,${entry.image_b64}" alt="${entry.label}">` : ''}
  `;

  const container = document.getElementById('log-container');
  if (prepend) container.prepend(div);
  else container.appendChild(div);

  // Max 300 DOM elem
  const entries = container.querySelectorAll('.log-entry');
  if (entries.length > 300) entries[entries.length-1].remove();
}

function updateStats() {
  document.getElementById('stat-total').textContent = allLogs.length;
  document.getElementById('stat-session').textContent = sessionCount;
  if (allLogs.length) {
    const avg = allLogs.reduce((s,e)=>s+e.confidence,0)/allLogs.length;
    document.getElementById('stat-conf').textContent = (avg*100).toFixed(1)+'%';
  }
}

function filterLogs() {
  filterTerm = document.getElementById('filter-input').value.toLowerCase();
  const container = document.getElementById('log-container');
  container.innerHTML = '';
  if (!allLogs.length) {
    container.innerHTML = '<div class="empty" id="empty-msg">Nincs találat.</div>';
    return;
  }
  [...allLogs].reverse().forEach(e => renderEntry(e, false));
}

function togglePause() {
  paused = !paused;
  document.getElementById('pause-btn').textContent = paused ? '▶ Folytatás' : '⏸ Pause';
}

function clearLogs() {
  allLogs = []; sessionCount = 0;
  document.getElementById('log-container').innerHTML =
    '<div class="empty" id="empty-msg">Log törölve.</div>';
  updateStats();
}

// Meglévő logok betöltése
fetch('/logs').then(r=>r.json()).then(data => {
  allLogs = [...data].reverse();
  allLogs.forEach(e => renderEntry(e, false));
  updateStats();
});

// SSE – live frissítés
const evtSource = new EventSource('/stream');
evtSource.onopen = () => {
  document.getElementById('dot').classList.add('live');
  document.getElementById('status-text').textContent = 'Élő';
};
evtSource.onerror = () => {
  document.getElementById('dot').classList.remove('live');
  document.getElementById('status-text').textContent = 'Szétkapcsolódott';
};
evtSource.onmessage = (e) => {
  if (paused) return;
  const entry = JSON.parse(e.data);
  allLogs.unshift(entry);
  if (allLogs.length > 500) allLogs.pop();
  sessionCount++;
  renderEntry(entry, true);
  updateStats();
};
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import socket
    ip = socket.gethostbyname(socket.gethostname())
    print(f"\n  Flask szerver fut: http://{ip}:5000")
    print(f"  Telefonról elérhető ugyanazon WiFi-n!\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
