# src/web_server.py
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os, threading, time

def create_app(shared_state, static_dir="../static"):
    app = Flask(__name__, static_folder=None)
    CORS(app)

    @app.route("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    @app.route("/api/live")
    def live():
        # 최근 샘플들만 반환
        with shared_state["lock"]:
            data = {
                "time":       list(shared_state["time"])[-180:],
                "gaze":       list(shared_state["gaze"])[-180:],
                "neck":       list(shared_state["neck"])[-180:],
                "emotion":    list(shared_state["emotion"])[-180:],
                "blink":      list(shared_state["blink"])[-180:],
                "focus":      list(shared_state["focus"])[-180:],
                "fps":        shared_state.get("fps", 0.0),
                "latest":     shared_state.get("latest", {})
            }
        return jsonify(data)

    @app.route("/api/session")
    def session():
        with shared_state["lock"]:
            meta = {
                "start_ts": shared_state.get("start_ts"),
                "frames":   shared_state.get("frames", 0),
                "saved":    shared_state.get("saved_rows", 0)
            }
        return jsonify(meta)
    return app

def run_server(app, host="127.0.0.1", port=8000):
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

