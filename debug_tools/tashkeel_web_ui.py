from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


UNSLOTH_PYTHON = r"C:\Users\Mohanad Dwekat\.unsloth\studio\unsloth_studio\Scripts\python.exe"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_V3_CATT_DIR = os.path.join(PROJECT_ROOT, "external", "catt")
DEFAULT_V4_CATT_DIR = os.path.join(PROJECT_ROOT, "external", "catt_punct_v4")
DEFAULT_V4_CHECKPOINT = os.path.join(
    DEFAULT_V4_CATT_DIR,
    "catt_eo_ft_balanced_v4_punct",
    "catt_eo_ft_best.pt",
)
DEFAULT_CATT_DIR = DEFAULT_V4_CATT_DIR if os.path.exists(DEFAULT_V4_CHECKPOINT) else DEFAULT_V3_CATT_DIR
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
HTML_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "tashkeel_web_ui.html")

sys.dont_write_bytecode = True


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tashkeel AI</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101417;
      --panel: #171c20;
      --panel-2: #1d2429;
      --ink: #f3efe6;
      --muted: #b8c0b8;
      --quiet: #778078;
      --line: #324047;
      --accent: #2fbf9b;
      --accent-2: #d6b25e;
      --danger: #e56b6f;
      --shadow: rgba(0, 0, 0, .28);
      --radius: 8px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 78% 18%, rgba(47, 191, 155, .10), transparent 24rem),
        linear-gradient(135deg, #101417 0%, #172025 42%, #111619 100%);
      color: var(--ink);
      font-family: "Segoe UI", Arial, sans-serif;
    }

    button, textarea, input { font: inherit; }

    .app {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto auto 1fr auto;
    }

    header {
      padding: 22px 28px 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      border-bottom: 1px solid rgba(255,255,255,.06);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
      min-width: 0;
    }

    .mark {
      width: 42px;
      height: 42px;
      border-radius: var(--radius);
      border: 1px solid rgba(255,255,255,.12);
      background: linear-gradient(160deg, #2fbf9b, #d6b25e);
      color: #101417;
      display: grid;
      place-items: center;
      font-family: Georgia, serif;
      font-size: 27px;
      font-weight: 700;
      box-shadow: 0 12px 28px var(--shadow);
      flex: 0 0 auto;
    }

    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.1;
      letter-spacing: 0;
    }

    .sub {
      margin-top: 5px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: min(66vw, 900px);
    }

    .status {
      display: flex;
      align-items: center;
      gap: 9px;
      padding: 9px 12px;
      border: 1px solid rgba(255,255,255,.10);
      background: rgba(23, 28, 32, .82);
      border-radius: var(--radius);
      color: var(--muted);
      font-size: 13px;
      min-width: 210px;
      justify-content: center;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: var(--quiet);
      box-shadow: 0 0 0 3px rgba(119, 128, 120, .15);
    }

    .dot.ready {
      background: var(--accent);
      box-shadow: 0 0 0 3px rgba(47, 191, 155, .16);
    }

    .toolbar {
      padding: 12px 28px;
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      border-bottom: 1px solid rgba(255,255,255,.06);
    }

    .btn {
      height: 38px;
      min-width: 38px;
      border: 1px solid rgba(255,255,255,.12);
      background: var(--panel-2);
      color: var(--ink);
      border-radius: var(--radius);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      cursor: pointer;
      padding: 0 12px;
      transition: transform .12s ease, background .12s ease, border-color .12s ease;
    }

    .btn:hover {
      transform: translateY(-1px);
      border-color: rgba(255,255,255,.24);
      background: #253037;
    }

    .btn:disabled {
      opacity: .55;
      cursor: not-allowed;
      transform: none;
    }

    .btn.primary {
      background: var(--accent);
      border-color: rgba(47, 191, 155, .8);
      color: #07110e;
      font-weight: 700;
    }

    .btn.primary:hover { background: #44d0ae; }

    .icon {
      width: 17px;
      height: 17px;
      display: inline-block;
      fill: currentColor;
      flex: 0 0 auto;
    }

    .split {
      flex: 1 1 auto;
      min-width: 12px;
    }

    .metric-strip {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    .metric {
      height: 38px;
      padding: 0 11px;
      display: flex;
      align-items: center;
      gap: 8px;
      border: 1px solid rgba(255,255,255,.08);
      border-radius: var(--radius);
      background: rgba(23, 28, 32, .72);
      color: var(--muted);
      font-size: 13px;
    }

    .metric strong {
      color: var(--ink);
      font-weight: 650;
    }

    main {
      padding: 18px 28px 20px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 16px;
      min-height: 0;
    }

    .panel {
      min-height: 0;
      display: grid;
      grid-template-rows: auto 1fr;
      border: 1px solid rgba(255,255,255,.10);
      background: rgba(23, 28, 32, .88);
      border-radius: var(--radius);
      box-shadow: 0 18px 40px var(--shadow);
      overflow: hidden;
    }

    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 13px;
      border-bottom: 1px solid rgba(255,255,255,.08);
      background: rgba(29, 36, 41, .74);
      min-height: 50px;
    }

    .panel-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 650;
    }

    .mini-actions {
      display: flex;
      gap: 6px;
    }

    .icon-btn {
      width: 32px;
      height: 32px;
      padding: 0;
    }

    textarea {
      width: 100%;
      height: 100%;
      resize: none;
      border: 0;
      outline: none;
      padding: 18px;
      background: transparent;
      color: var(--ink);
      font-family: Tahoma, "Segoe UI", Arial, sans-serif;
      font-size: 20px;
      line-height: 1.95;
      direction: rtl;
      text-align: right;
      letter-spacing: 0;
    }

    textarea::placeholder {
      color: #6f7772;
    }

    .output {
      white-space: pre-wrap;
      overflow: auto;
      padding: 18px;
      font-family: Tahoma, "Segoe UI", Arial, sans-serif;
      font-size: 20px;
      line-height: 1.95;
      direction: rtl;
      text-align: right;
      color: var(--ink);
    }

    .placeholder {
      color: #6f7772;
    }

    footer {
      padding: 0 28px 18px;
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }

    .telemetry {
      border: 1px solid rgba(255,255,255,.08);
      background: rgba(23, 28, 32, .72);
      border-radius: var(--radius);
      padding: 11px 12px;
      color: var(--muted);
      font-size: 13px;
      min-height: 42px;
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }

    .warn {
      color: #f0cf8a;
    }

    .error {
      color: var(--danger);
    }

    @media (max-width: 880px) {
      header {
        align-items: flex-start;
        flex-direction: column;
        padding: 18px 16px 10px;
      }

      .status {
        width: 100%;
        justify-content: flex-start;
      }

      .toolbar {
        padding: 12px 16px;
      }

      main {
        grid-template-columns: 1fr;
        padding: 14px 16px 16px;
        min-height: 780px;
      }

      footer {
        padding: 0 16px 16px;
      }

      textarea, .output {
        font-size: 18px;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="brand">
        <div class="mark" aria-hidden="true">ض</div>
        <div>
          <h1>Tashkeel AI</h1>
          <div class="sub" id="checkpoint">Loading model status...</div>
        </div>
      </div>
      <div class="status" title="Model runtime status">
        <span class="dot" id="statusDot"></span>
        <span id="statusText">Starting</span>
      </div>
    </header>

    <section class="toolbar" aria-label="Actions">
      <button class="btn primary" id="runBtn" title="Run diacritization">
        <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>
        Diacritize
      </button>
      <button class="btn" id="sampleBtn" title="Load a short sample">
        <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 4h16v2H4zm0 4h10v2H4zm0 4h16v2H4zm0 4h10v2H4z"/></svg>
        Sample
      </button>
      <button class="btn" id="clearBtn" title="Clear input and output">
        <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6zm3-9h2v8H9zm4 0h2v8h-2zM15.5 4l-1-1h-5l-1 1H5v2h14V4z"/></svg>
        Clear
      </button>
      <div class="split"></div>
      <div class="metric-strip">
        <div class="metric">Chars <strong id="charCount">0</strong></div>
        <div class="metric">Arabic <strong id="arabicCount">0</strong></div>
        <div class="metric">Time <strong id="elapsed">n/a</strong></div>
      </div>
    </section>

    <main>
      <section class="panel">
        <div class="panel-head">
          <div class="panel-title">
            <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 17.25V21h3.75L17.8 9.95l-3.75-3.75zM20.7 7.05c.4-.4.4-1 0-1.4l-2.35-2.35a.99.99 0 0 0-1.4 0l-1.85 1.85 3.75 3.75z"/></svg>
            Input
          </div>
          <div class="mini-actions">
            <button class="btn icon-btn" id="pasteBtn" title="Paste from clipboard">
              <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M19 3h-4.2C14.4 1.8 13.3 1 12 1S9.6 1.8 9.2 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2m-7 0c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1m7 16H5V5h2v3h10V5h2z"/></svg>
            </button>
          </div>
        </div>
        <textarea id="inputText" spellcheck="false" placeholder="الصق النص العربي غير المشكول هنا"></textarea>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div class="panel-title">
            <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z"/></svg>
            Output
          </div>
          <div class="mini-actions">
            <button class="btn icon-btn" id="copyBtn" title="Copy output">
              <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M16 1H4c-1.1 0-2 .9-2 2v12h2V3h12zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2m0 16H8V7h11z"/></svg>
            </button>
          </div>
        </div>
        <div class="output placeholder" id="outputText">الناتج سيظهر هنا</div>
      </section>
    </main>

    <footer>
      <div class="telemetry" id="telemetry">Ready for text.</div>
    </footer>
  </div>

  <script>
    const input = document.getElementById('inputText');
    const output = document.getElementById('outputText');
    const runBtn = document.getElementById('runBtn');
    const clearBtn = document.getElementById('clearBtn');
    const sampleBtn = document.getElementById('sampleBtn');
    const pasteBtn = document.getElementById('pasteBtn');
    const copyBtn = document.getElementById('copyBtn');
    const checkpoint = document.getElementById('checkpoint');
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    const telemetry = document.getElementById('telemetry');
    const charCount = document.getElementById('charCount');
    const arabicCount = document.getElementById('arabicCount');
    const elapsed = document.getElementById('elapsed');

    const sampleText = 'قال الطالب إن المشروع انتهى بنجاح، لكننا نحتاج إلى واجهة بسيطة للتجربة.\nاكتب هنا أي نص عربي بدون تشكيل ثم اضغط زر إضافة التشكيل.';

    function setBusy(isBusy) {
      runBtn.disabled = isBusy;
      runBtn.innerHTML = isBusy
        ? '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2a10 10 0 1 0 10 10h-2a8 8 0 1 1-8-8z"/></svg> Running'
        : '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg> Diacritize';
    }

    function updateCounts() {
      const text = input.value;
      charCount.textContent = String(text.length);
      const matches = text.match(/[\u0621-\u064A\u0671]/g);
      arabicCount.textContent = String(matches ? matches.length : 0);
    }

    function setOutput(text) {
      output.textContent = text || 'الناتج سيظهر هنا';
      output.classList.toggle('placeholder', !text);
    }

    async function getStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        checkpoint.textContent = data.checkpoint;
        statusText.textContent = data.loaded ? `Loaded on ${data.device}` : 'Ready';
        statusDot.classList.toggle('ready', data.loaded);
      } catch (error) {
        statusText.textContent = 'Backend unavailable';
        telemetry.innerHTML = `<span class="error">${error}</span>`;
      }
    }

    async function runTashkeel() {
      const text = input.value.trim();
      if (!text) {
        telemetry.innerHTML = '<span class="warn">Paste Arabic text first.</span>';
        return;
      }

      setBusy(true);
      telemetry.textContent = 'Running inference... first run may take a few seconds while the checkpoint loads.';
      const started = performance.now();

      try {
        const res = await fetch('/api/diacritize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          throw new Error(data.error || `HTTP ${res.status}`);
        }
        setOutput(data.output);
        elapsed.textContent = `${data.elapsed.toFixed(2)}s`;
        statusText.textContent = `Loaded on ${data.device}`;
        statusDot.classList.add('ready');

        const warnings = data.warnings && data.warnings.length
          ? ` <span class="warn">${data.warnings.join(' ')}</span>`
          : '';
        telemetry.innerHTML =
          `Segments <strong>${data.segments}</strong> · Chunked <strong>${data.chunked}</strong> · ` +
          `Backend <strong>${data.elapsed.toFixed(2)}s</strong> · Wall <strong>${((performance.now() - started) / 1000).toFixed(2)}s</strong>` +
          warnings;
      } catch (error) {
        telemetry.innerHTML = `<span class="error">${error.message}</span>`;
      } finally {
        setBusy(false);
      }
    }

    input.addEventListener('input', updateCounts);
    runBtn.addEventListener('click', runTashkeel);
    input.addEventListener('keydown', (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        runTashkeel();
      }
    });
    clearBtn.addEventListener('click', () => {
      input.value = '';
      setOutput('');
      elapsed.textContent = 'n/a';
      telemetry.textContent = 'Ready for text.';
      updateCounts();
      input.focus();
    });
    sampleBtn.addEventListener('click', () => {
      input.value = sampleText;
      updateCounts();
      input.focus();
    });
    pasteBtn.addEventListener('click', async () => {
      try {
        input.value = await navigator.clipboard.readText();
        updateCounts();
      } catch (error) {
        telemetry.innerHTML = `<span class="warn">Clipboard paste was blocked by the browser.</span>`;
      }
    });
    copyBtn.addEventListener('click', async () => {
      const text = output.classList.contains('placeholder') ? '' : output.textContent;
      if (!text) return;
      await navigator.clipboard.writeText(text);
      telemetry.textContent = 'Output copied.';
    });

    updateCounts();
    getStatus();
  </script>
</body>
</html>
"""


def ensure_utf8_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def relaunch_with_project_python() -> None:
    current_python = os.path.normcase(os.path.abspath(sys.executable))
    project_python = os.path.normcase(os.path.abspath(UNSLOTH_PYTHON))
    if current_python != project_python and os.path.exists(UNSLOTH_PYTHON):
        raise SystemExit(
            subprocess.call([UNSLOTH_PYTHON, os.path.abspath(__file__), *sys.argv[1:]])
        )


def find_free_port(host: str, preferred_port: int) -> int:
    for port in range(preferred_port, preferred_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free port found starting at {preferred_port}")


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def load_html_page() -> str:
    try:
        with open(HTML_TEMPLATE_PATH, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return HTML_PAGE


class TashkeelWebApp:
    def __init__(
        self,
        catt_dir: str = DEFAULT_CATT_DIR,
        checkpoint: str | None = None,
        model: str = "eo",
        batch_size: int = 32,
        max_seq_len: int = 1024,
    ) -> None:
        catt_dir = os.path.abspath(catt_dir)
        if catt_dir == os.path.abspath(DEFAULT_V4_CATT_DIR):
            if catt_dir not in sys.path:
                sys.path.insert(0, catt_dir)
            from local_inference import LocalCATTInference

            self.inferencer = LocalCATTInference(
                checkpoint=checkpoint,
                model_type=model,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
            )
        else:
            from tashkeel_ui import LocalCATTInference

            self.inferencer = LocalCATTInference(
                catt_dir=catt_dir,
                checkpoint=checkpoint,
                model_type=model,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
            )
        self.lock = threading.Lock()

    def status(self) -> dict:
        return {
            "checkpoint": self.inferencer.checkpoint,
            "loaded": self.inferencer.model is not None,
            "device": self.inferencer.device,
            "model": self.inferencer.model_type,
        }

    def diacritize(self, text: str) -> dict:
        if not text.strip():
            raise ValueError("Input text is empty.")

        with self.lock:
            output, telemetry = self.inferencer.diacritize(text)

        segment_info = telemetry["segment_info"]
        return {
            "output": output,
            "elapsed": telemetry["elapsed"],
            "segments": segment_info.inferred_segments,
            "chunked": segment_info.chunked_segments,
            "warnings": segment_info.warnings,
            "device": self.inferencer.device,
            "checkpoint": self.inferencer.checkpoint,
        }


def make_handler(app: TashkeelWebApp):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = load_html_page().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/status":
                json_response(self, 200, app.status())
                return
            json_response(self, 404, {"error": "Not found"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/diacritize":
                json_response(self, 404, {"error": "Not found"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length)
                payload = json.loads(raw_body.decode("utf-8") or "{}")
                text = str(payload.get("text", ""))
                json_response(self, 200, app.diacritize(text))
            except Exception as exc:
                traceback.print_exc()
                json_response(self, 500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt: str, *args) -> None:
            print(f"[web] {self.address_string()} - {fmt % args}")

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--catt-dir", default=DEFAULT_CATT_DIR)
    parser.add_argument("--model", choices=["eo", "ed"], default="eo")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    relaunch_with_project_python()
    ensure_utf8_console()
    args = parse_args()

    app = TashkeelWebApp(
        catt_dir=args.catt_dir,
        checkpoint=args.checkpoint,
        model=args.model,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )
    port = find_free_port(args.host, args.port)
    server = ThreadingHTTPServer((args.host, port), make_handler(app))
    url = f"http://{args.host}:{port}/"

    print("Tashkeel AI")
    print(f"URL:        {url}")
    print(f"Checkpoint: {app.inferencer.checkpoint}")
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
