from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V4_CATT_DIR = os.path.join(PROJECT_ROOT, "external", "catt_punct_v4")
HTML_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "tashkeel_web_ui.html")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8865


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
            sock.settimeout(0.2)
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise OSError(f"No free port found from {preferred_port} to {preferred_port + 49}")


def load_html_page() -> str:
    with open(HTML_TEMPLATE_PATH, "r", encoding="utf-8") as handle:
        return handle.read()


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


class TashkeelWebApp:
    def __init__(self, checkpoint: str | None = None, batch_size: int = 32, max_seq_len: int = 1024) -> None:
        if V4_CATT_DIR not in sys.path:
            sys.path.insert(0, V4_CATT_DIR)
        from local_inference import LocalCATTInference

        self.inferencer = LocalCATTInference(
            checkpoint=checkpoint,
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
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                text = str(payload.get("text", ""))
                json_response(self, 200, app.diacritize(text))
            except Exception as exc:
                traceback.print_exc()
                json_response(self, 500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt: str, *args) -> None:
            print(f"[v4-web] {self.address_string()} - {fmt % args}")

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--checkpoint")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    ensure_utf8_console()
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    app = TashkeelWebApp(
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )
    port = find_free_port(args.host, args.port)
    server = ThreadingHTTPServer((args.host, port), make_handler(app))
    url = f"http://{args.host}:{port}/"

    print("Tashkeel AI v4 punctuation-aware")
    print(f"URL:        {url}")
    print(f"Checkpoint: {app.inferencer.checkpoint}")
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
