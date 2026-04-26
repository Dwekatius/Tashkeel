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
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


UNSLOTH_PYTHON = r"C:\Users\Mohanad Dwekat\.unsloth\studio\unsloth_studio\Scripts\python.exe"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ML_SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "ML_Tashkeel", "scripts")
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "ML_Tashkeel", "models", "punctuation_restorer")
CHALLENGER_MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "ML_Tashkeel",
    "models",
    "punctuation_restorer_own_v2_balanced_bs32_cw8",
)
DEFAULT_THRESHOLDS = "COMMA=0.95,QUESTION=0.9,COLON=0.98,SEMICOLON=0.93,EXCLAMATION=0.98"
CHALLENGER_THRESHOLDS = "COMMA=0.95,PERIOD=0.5,QUESTION=0.97,COLON=0.97,SEMICOLON=0.9,EXCLAMATION=0.99"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8776

sys.dont_write_bytecode = True


def relaunch_with_project_python() -> None:
    current_python = os.path.normcase(os.path.abspath(sys.executable))
    project_python = os.path.normcase(os.path.abspath(UNSLOTH_PYTHON))
    if current_python != project_python and os.path.exists(UNSLOTH_PYTHON):
        raise SystemExit(
            subprocess.call([UNSLOTH_PYTHON, os.path.abspath(__file__), *sys.argv[1:]])
        )


if __name__ == "__main__":
    relaunch_with_project_python()


if ML_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, ML_SCRIPTS_DIR)

from punct_common import LABEL_TO_OUTPUT, extract_words_and_labels, reconstruct_text, strip_diacritics
from punct_modeling import choose_device, load_model, load_tokenizer, predict_labels_for_words
from punctuate_text import parse_label_thresholds


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Punctuation Restorer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111417;
      --panel: #171d21;
      --panel-2: #20282d;
      --ink: #f4f1e8;
      --muted: #b9c0b7;
      --quiet: #7c867d;
      --line: #344047;
      --accent: #35bea2;
      --accent-2: #d8b55f;
      --danger: #e37070;
      --radius: 8px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: #111417;
      color: var(--ink);
      font-family: "Segoe UI", Arial, sans-serif;
    }

    button, textarea, input, select {
      font: inherit;
    }

    .app {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto auto 1fr;
    }

    header {
      padding: 18px 24px 12px;
      border-bottom: 1px solid rgba(255,255,255,.07);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }

    h1 {
      margin: 0;
      font-size: 21px;
      line-height: 1.1;
      letter-spacing: 0;
    }

    .sub {
      margin-top: 5px;
      color: var(--muted);
      font-size: 13px;
    }

    .status {
      min-width: 220px;
      border: 1px solid rgba(255,255,255,.11);
      border-radius: var(--radius);
      background: var(--panel);
      padding: 9px 12px;
      color: var(--muted);
      font-size: 13px;
      text-align: center;
    }

    .toolbar {
      padding: 12px 24px;
      border-bottom: 1px solid rgba(255,255,255,.07);
      display: grid;
      grid-template-columns: minmax(190px, 260px) minmax(260px, 1fr) minmax(320px, 1.5fr);
      gap: 10px;
      align-items: end;
      background: #13181b;
    }

    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
    }

    input, select, textarea {
      width: 100%;
      border: 1px solid rgba(255,255,255,.12);
      border-radius: var(--radius);
      background: var(--panel);
      color: var(--ink);
      outline: none;
    }

    input, select {
      height: 38px;
      padding: 0 10px;
    }

    textarea {
      min-height: 340px;
      padding: 14px;
      line-height: 1.9;
      resize: vertical;
      direction: rtl;
      unicode-bidi: plaintext;
      font-size: 20px;
    }

    input:focus, select:focus, textarea:focus {
      border-color: rgba(53,190,162,.72);
      box-shadow: 0 0 0 3px rgba(53,190,162,.13);
    }

    .options {
      padding: 10px 24px 0;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }

    .check {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      min-height: 38px;
    }

    .check input {
      width: 16px;
      height: 16px;
    }

    .btn {
      height: 38px;
      border: 1px solid rgba(255,255,255,.14);
      border-radius: var(--radius);
      background: var(--panel-2);
      color: var(--ink);
      padding: 0 14px;
      cursor: pointer;
    }

    .btn:hover { border-color: rgba(255,255,255,.28); }
    .btn.primary {
      background: var(--accent);
      color: #071311;
      border-color: transparent;
      font-weight: 700;
    }

    .btn:disabled {
      opacity: .55;
      cursor: wait;
    }

    main {
      padding: 16px 24px 22px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      min-height: 0;
    }

    .pane {
      min-width: 0;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 8px;
    }

    .pane-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
    }

    .metrics {
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 8px;
    }

    .metric {
      border: 1px solid rgba(255,255,255,.10);
      border-radius: var(--radius);
      background: var(--panel);
      padding: 9px 10px;
      min-height: 58px;
    }

    .metric span {
      display: block;
      color: var(--quiet);
      font-size: 11px;
      margin-bottom: 5px;
    }

    .metric strong {
      font-size: 18px;
      letter-spacing: 0;
    }

    .marks {
      grid-column: 1 / -1;
      border: 1px solid rgba(255,255,255,.10);
      border-radius: var(--radius);
      background: var(--panel);
      min-height: 88px;
      max-height: 210px;
      overflow: auto;
      padding: 10px;
      color: var(--muted);
      font-size: 13px;
    }

    .mark-row {
      display: grid;
      grid-template-columns: 72px 1fr 110px 90px;
      gap: 10px;
      padding: 6px 0;
      border-bottom: 1px solid rgba(255,255,255,.06);
      align-items: center;
    }

    .mark-row:last-child { border-bottom: 0; }
    .word { direction: rtl; unicode-bidi: plaintext; color: var(--ink); }
    .error { color: var(--danger); }

    @media (max-width: 900px) {
      header { align-items: flex-start; flex-direction: column; }
      .status { width: 100%; }
      .toolbar { grid-template-columns: 1fr; }
      main { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .mark-row { grid-template-columns: 56px 1fr 86px; }
      .mark-row .source { display: none; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div>
        <h1>Punctuation Restorer</h1>
        <div class="sub">Arabic punctuation only</div>
      </div>
      <div class="status" id="status">loading model info...</div>
    </header>

    <section class="toolbar">
      <label>
        Preset
        <select id="preset">
          <option value="official">Official baseline - calibrated</option>
          <option value="baseline_raw">Official baseline - raw</option>
          <option value="balanced_raw">Balanced own_v2 - raw</option>
          <option value="balanced_tuned">Balanced own_v2 - tuned</option>
          <option value="custom">Custom</option>
        </select>
      </label>
      <label>
        Model dir
        <input id="modelDir" spellcheck="false" />
      </label>
      <label>
        Label thresholds
        <input id="thresholds" spellcheck="false" />
      </label>
    </section>

    <section class="options">
      <button class="btn primary" id="runBtn">Punctuate</button>
      <button class="btn" id="sampleBtn">Sample</button>
      <button class="btn" id="clearBtn">Clear</button>
      <label class="check"><input id="preserveExisting" type="checkbox" checked /> Preserve existing punctuation</label>
      <label class="check"><input id="showMarks" type="checkbox" checked /> Show marks</label>
    </section>

    <main>
      <div class="pane">
        <div class="pane-head"><span>Input</span><span id="inputCount">0 chars</span></div>
        <textarea id="input" placeholder="ضع النص العربي هنا"></textarea>
      </div>
      <div class="pane">
        <div class="pane-head"><span>Output</span><span id="outputCount">0 chars</span></div>
        <textarea id="output" readonly></textarea>
      </div>

      <section class="metrics" id="metrics"></section>
      <section class="marks" id="marks">No output yet.</section>
    </main>
  </div>

  <script>
    const presets = {
      official: {
        modelDir: "ML_Tashkeel\\models\\punctuation_restorer",
        thresholds: "COMMA=0.95,QUESTION=0.9,COLON=0.98,SEMICOLON=0.93,EXCLAMATION=0.98"
      },
      baseline_raw: {
        modelDir: "ML_Tashkeel\\models\\punctuation_restorer",
        thresholds: ""
      },
      balanced_raw: {
        modelDir: "ML_Tashkeel\\models\\punctuation_restorer_own_v2_balanced_bs32_cw8",
        thresholds: ""
      },
      balanced_tuned: {
        modelDir: "ML_Tashkeel\\models\\punctuation_restorer_own_v2_balanced_bs32_cw8",
        thresholds: "COMMA=0.95,PERIOD=0.5,QUESTION=0.97,COLON=0.97,SEMICOLON=0.9,EXCLAMATION=0.99"
      }
    };

    const sampleText = "قالت ليلى هل عاد سامر من الجامعة اليوم ثم أضافت لا أظن أنه تأخر بلا سبب";
    const el = (id) => document.getElementById(id);

    function setPreset(name) {
      if (!presets[name]) return;
      el("modelDir").value = presets[name].modelDir;
      el("thresholds").value = presets[name].thresholds;
    }

    function setStatus(text, isError = false) {
      el("status").textContent = text;
      el("status").className = isError ? "status error" : "status";
    }

    function updateCounts() {
      el("inputCount").textContent = `${el("input").value.length} chars`;
      el("outputCount").textContent = `${el("output").value.length} chars`;
    }

    function metric(label, value) {
      return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
    }

    function renderMetrics(data) {
      el("metrics").innerHTML = [
        metric("device", data.device || "-"),
        metric("elapsed", `${(data.elapsed || 0).toFixed(3)}s`),
        metric("boundaries", data.word_boundaries || 0),
        metric("predicted", data.predicted_punct || 0),
        metric("final", data.final_punct || 0),
        metric("model", data.model_name || "-")
      ].join("");
    }

    function renderMarks(marks) {
      if (!el("showMarks").checked) {
        el("marks").textContent = "Marks hidden.";
        return;
      }
      if (!marks || marks.length === 0) {
        el("marks").textContent = "No punctuation marks in output.";
        return;
      }
      const rows = marks.slice(0, 350).map((m) => `
        <div class="mark-row">
          <div>#${m.index + 1}</div>
          <div class="word">${m.word}</div>
          <div>${m.label}</div>
          <div class="source">${m.source}</div>
        </div>
      `).join("");
      const more = marks.length > 350 ? `<div class="mark-row"><div></div><div>${marks.length - 350} more...</div><div></div><div></div></div>` : "";
      el("marks").innerHTML = rows + more;
    }

    async function getStatus() {
      try {
        const res = await fetch("/api/status");
        const data = await res.json();
        setStatus(`ready - ${data.device || "unknown"} - ${data.default_model_name}`);
        renderMetrics({
          device: data.device || "-",
          elapsed: 0,
          word_boundaries: 0,
          predicted_punct: 0,
          final_punct: 0,
          model_name: data.default_model_name || "-"
        });
      } catch (err) {
        setStatus(`status failed: ${err.message}`, true);
      }
    }

    async function punctuate() {
      const text = el("input").value;
      if (!text.trim()) return;
      el("runBtn").disabled = true;
      setStatus("running...");
      try {
        const res = await fetch("/api/punctuate", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            text,
            model_dir: el("modelDir").value,
            label_thresholds: el("thresholds").value,
            preserve_existing: el("preserveExisting").checked
          })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "request failed");
        el("output").value = data.output || "";
        updateCounts();
        renderMetrics(data);
        renderMarks(data.marks);
        setStatus(`ready - ${data.device} - ${data.model_name}`);
      } catch (err) {
        setStatus(err.message, true);
        el("marks").textContent = err.message;
      } finally {
        el("runBtn").disabled = false;
      }
    }

    el("preset").addEventListener("change", (event) => setPreset(event.target.value));
    el("runBtn").addEventListener("click", punctuate);
    el("sampleBtn").addEventListener("click", () => {
      el("input").value = sampleText;
      updateCounts();
    });
    el("clearBtn").addEventListener("click", () => {
      el("input").value = "";
      el("output").value = "";
      el("marks").textContent = "No output yet.";
      updateCounts();
    });
    el("input").addEventListener("input", updateCounts);
    el("showMarks").addEventListener("change", () => {
      if (!el("output").value) el("marks").textContent = "No output yet.";
    });

    setPreset("official");
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


def resolve_model_dir(value: str) -> str:
    model_dir = value.strip() or DEFAULT_MODEL_DIR
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(PROJECT_ROOT, model_dir)
    return os.path.abspath(model_dir)


def short_model_name(path: str) -> str:
    return os.path.basename(os.path.normpath(path))


class PunctuationWebApp:
    def __init__(
        self,
        default_model_dir: str = DEFAULT_MODEL_DIR,
        max_length: int = 256,
        max_words: int = 180,
        batch_size: int = 32,
    ) -> None:
        self.default_model_dir = resolve_model_dir(default_model_dir)
        self.max_length = max_length
        self.max_words = max_words
        self.batch_size = batch_size
        self.device = choose_device()
        self.lock = threading.Lock()
        self.cache: dict[str, tuple[object, object]] = {}
        self.load_model_bundle(self.default_model_dir)

    def load_model_bundle(self, model_dir: str):
        model_dir = resolve_model_dir(model_dir)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model dir not found: {model_dir}")
        if model_dir not in self.cache:
            tokenizer = load_tokenizer(model_dir)
            model = load_model(model_dir)
            model.to(self.device)
            model.eval()
            self.cache[model_dir] = (tokenizer, model)
        return self.cache[model_dir]

    def status(self) -> dict:
        return {
            "device": str(self.device),
            "default_model_dir": self.default_model_dir,
            "default_model_name": short_model_name(self.default_model_dir),
            "loaded_models": [short_model_name(path) for path in self.cache],
            "default_thresholds": DEFAULT_THRESHOLDS,
            "challenger_model_dir": CHALLENGER_MODEL_DIR,
            "challenger_thresholds": CHALLENGER_THRESHOLDS,
        }

    def punctuate(
        self,
        text: str,
        model_dir: str,
        label_thresholds: str,
        preserve_existing: bool = True,
    ) -> dict:
        if not text.strip():
            raise ValueError("Input text is empty.")

        thresholds = parse_label_thresholds(label_thresholds)
        model_dir = resolve_model_dir(model_dir)
        start = time.time()

        with self.lock:
            tokenizer, model = self.load_model_bundle(model_dir)
            example = extract_words_and_labels(strip_diacritics(text), source="input")
            if example is None:
                return {
                    "output": text,
                    "elapsed": time.time() - start,
                    "device": str(self.device),
                    "model_dir": model_dir,
                    "model_name": short_model_name(model_dir),
                    "word_boundaries": 0,
                    "predicted_punct": 0,
                    "final_punct": 0,
                    "label_counts": {},
                    "marks": [],
                }

            predicted_labels, confidences = predict_labels_for_words(
                model=model,
                tokenizer=tokenizer,
                words=example.words,
                max_length=self.max_length,
                max_words=self.max_words,
                batch_size=self.batch_size,
                label_thresholds=thresholds,
                device=self.device,
            )

        final_labels: list[str] = []
        marks: list[dict] = []
        for idx, (word, existing, predicted, confidence) in enumerate(
            zip(example.words, example.labels, predicted_labels, confidences)
        ):
            if preserve_existing and existing != "NONE":
                label = existing
                source = "existing"
            else:
                label = predicted
                source = "predicted"
            final_labels.append(label)
            if label != "NONE":
                marks.append(
                    {
                        "index": idx,
                        "word": word,
                        "label": label,
                        "mark": LABEL_TO_OUTPUT.get(label, ""),
                        "confidence": confidence,
                        "source": source,
                    }
                )

        output = reconstruct_text(example.words, final_labels)
        label_counts = Counter(final_labels)
        predicted_count = sum(1 for label in predicted_labels if label != "NONE")
        final_count = sum(1 for label in final_labels if label != "NONE")

        return {
            "output": output,
            "elapsed": time.time() - start,
            "device": str(self.device),
            "model_dir": model_dir,
            "model_name": short_model_name(model_dir),
            "word_boundaries": len(final_labels),
            "predicted_punct": predicted_count,
            "final_punct": final_count,
            "label_counts": dict(label_counts),
            "marks": marks,
        }


def make_handler(app: PunctuationWebApp):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = HTML_PAGE.encode("utf-8")
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
            if parsed.path != "/api/punctuate":
                json_response(self, 404, {"error": "Not found"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length)
                payload = json.loads(raw_body.decode("utf-8") or "{}")
                json_response(
                    self,
                    200,
                    app.punctuate(
                        text=str(payload.get("text", "")),
                        model_dir=str(payload.get("model_dir", "")),
                        label_thresholds=str(payload.get("label_thresholds", "")),
                        preserve_existing=bool(payload.get("preserve_existing", True)),
                    ),
                )
            except Exception as exc:
                traceback.print_exc()
                json_response(self, 500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt: str, *args) -> None:
            print(f"[punct-ui] {self.address_string()} - {fmt % args}")

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local web UI for Arabic punctuation restoration.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-words", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    relaunch_with_project_python()
    ensure_utf8_console()
    os.chdir(PROJECT_ROOT)
    args = parse_args()

    app = PunctuationWebApp(
        default_model_dir=args.model_dir,
        max_length=args.max_length,
        max_words=args.max_words,
        batch_size=args.batch_size,
    )
    port = find_free_port(args.host, args.port)
    server = ThreadingHTTPServer((args.host, port), make_handler(app))
    url = f"http://{args.host}:{port}/"

    print("Punctuation Restorer UI")
    print(f"URL:   {url}")
    print(f"Model: {app.default_model_dir}")
    print(f"Device: {app.device}")
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
