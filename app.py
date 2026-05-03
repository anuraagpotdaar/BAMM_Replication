"""Flask server for BAMM real-time motion preview.
Mirrors the FloodDiffusion API contract so the Three.js UI plugs in unchanged.
"""

import argparse
import threading
import time

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from model_manager import get_model_manager


def _coerce_value(value, reference):
    if isinstance(reference, bool):
        return value if isinstance(value, bool) else str(value).lower() in ("true", "1")
    elif isinstance(reference, int):
        return int(value)
    elif isinstance(reference, float):
        return float(value)
    return str(value)


app = Flask(__name__)
CORS(app)

model_manager = None
active_session_id = None
session_lock = threading.Lock()
PERF_OUT_DIR = "."  # overridden by --perf-dir at startup

last_frame_consumed_time = None
consumption_timeout = 60.0
consumption_monitor_thread = None
consumption_monitor_lock = threading.Lock()


def init_model():
    global model_manager
    if model_manager is None:
        print("[BAMM] initializing model manager")
        model_manager = get_model_manager()
        print("[BAMM] model manager ready")
    return model_manager


def consumption_monitor():
    global last_frame_consumed_time, active_session_id, model_manager
    while True:
        time.sleep(2.0)
        should_reset = False
        current_session = None
        time_since_last = 0
        with consumption_monitor_lock:
            if last_frame_consumed_time is not None:
                time_since_last = time.time() - last_frame_consumed_time
                if time_since_last > consumption_timeout:
                    if model_manager and model_manager.is_generating:
                        should_reset = True
        if should_reset:
            with session_lock:
                current_session = active_session_id
        if should_reset and current_session is not None:
            print(f"[BAMM] no frame consumed for {time_since_last:.1f}s, auto-reset")
            if model_manager:
                model_manager.reset()
            with session_lock:
                if active_session_id == current_session:
                    active_session_id = None
            with consumption_monitor_lock:
                last_frame_consumed_time = None


def start_consumption_monitor():
    global consumption_monitor_thread
    if consumption_monitor_thread is None or not consumption_monitor_thread.is_alive():
        consumption_monitor_thread = threading.Thread(
            target=consumption_monitor, daemon=True,
        )
        consumption_monitor_thread.start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config", methods=["GET"])
def get_config():
    try:
        if model_manager:
            status = model_manager.get_buffer_status()
            return jsonify({
                "schedule_config": status["schedule_config"],
                "cfg_config": status["cfg_config"],
                "history_length": status["history_length"],
                "smoothing_alpha": float(status["smoothing_alpha"]),
            })
        return jsonify({
            "schedule_config": {}, "cfg_config": {},
            "history_length": 30, "smoothing_alpha": 1.0,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/config", methods=["POST"])
def update_config():
    global active_session_id, last_frame_consumed_time
    try:
        if not model_manager:
            return jsonify({"status": "error", "message": "Model not loaded"}), 400

        data = request.json
        new_schedule = data.get("schedule_config")
        new_cfg = data.get("cfg_config")
        history_length = data.get("history_length")
        smoothing_alpha = data.get("smoothing_alpha")

        valid_schedule = set(model_manager._base_schedule_config.keys())
        valid_cfg = set(model_manager._base_cfg_config.keys())

        if new_schedule:
            for k in new_schedule:
                if k not in valid_schedule:
                    return jsonify({"status": "error", "message": f"Unknown schedule key: {k}"}), 400
            for k, v in new_schedule.items():
                model_manager._base_schedule_config[k] = _coerce_value(
                    v, model_manager._base_schedule_config[k],
                )
        if new_cfg:
            for k in new_cfg:
                if k not in valid_cfg:
                    return jsonify({"status": "error", "message": f"Unknown cfg key: {k}"}), 400
            for k, v in new_cfg.items():
                model_manager._base_cfg_config[k] = _coerce_value(
                    v, model_manager._base_cfg_config[k],
                )

        model_manager.reset(history_length=history_length, smoothing_alpha=smoothing_alpha)

        with session_lock:
            active_session_id = None
        with consumption_monitor_lock:
            last_frame_consumed_time = None

        return jsonify({"status": "success"})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/start", methods=["POST"])
def start_generation():
    global active_session_id, last_frame_consumed_time
    try:
        data = request.json
        session_id = data.get("session_id")
        text = data.get("text", "walk forward")
        history_length = data.get("history_length")
        smoothing_alpha = data.get("smoothing_alpha")
        force = data.get("force", False)

        if not session_id:
            return jsonify({"status": "error", "message": "session_id is required"}), 400

        mm = init_model()

        need_force_takeover = False
        with session_lock:
            if active_session_id and active_session_id != session_id:
                if not force:
                    return jsonify({
                        "status": "error",
                        "message": "Another session is already generating.",
                        "conflict": True,
                        "active_session_id": active_session_id,
                    }), 409
                need_force_takeover = True
            if mm.is_generating and active_session_id == session_id:
                return jsonify({
                    "status": "error",
                    "message": "Generation is already running for this session.",
                }), 400
            active_session_id = session_id

        if need_force_takeover:
            with consumption_monitor_lock:
                last_frame_consumed_time = None

        mm.reset(history_length=history_length, smoothing_alpha=smoothing_alpha)
        mm.start_generation(text, history_length=history_length)

        with consumption_monitor_lock:
            last_frame_consumed_time = time.time()
        start_consumption_monitor()

        return jsonify({
            "status": "success",
            "message": f"Generation started: {text}",
            "session_id": session_id,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/update_text", methods=["POST"])
def update_text():
    try:
        data = request.json
        session_id = data.get("session_id")
        text = data.get("text", "")
        if not session_id:
            return jsonify({"status": "error", "message": "session_id is required"}), 400
        with session_lock:
            if active_session_id != session_id:
                return jsonify({"status": "error", "message": "Not the active session"}), 403
        if model_manager is None:
            return jsonify({"status": "error", "message": "Model not initialized"}), 400
        model_manager.update_text(text)
        return jsonify({"status": "success", "message": f"Text updated: {text}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/telemetry", methods=["POST"])
def client_telemetry():
    return jsonify({"status": "ok"})  # no-op


@app.route("/api/pause", methods=["POST"])
def pause_generation():
    try:
        data = request.json or {}
        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"status": "error", "message": "session_id is required"}), 400
        with session_lock:
            if active_session_id != session_id:
                return jsonify({"status": "error", "message": "Not the active session"}), 403
        if model_manager:
            model_manager.pause_generation()
        return jsonify({"status": "success", "message": "Generation paused"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/resume", methods=["POST"])
def resume_generation():
    global last_frame_consumed_time
    try:
        data = request.json or {}
        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"status": "error", "message": "session_id is required"}), 400
        with session_lock:
            if active_session_id != session_id:
                return jsonify({"status": "error", "message": "Not the active session"}), 403
        if model_manager is None:
            return jsonify({"status": "error", "message": "Model not initialized"}), 400
        model_manager.resume_generation()
        with consumption_monitor_lock:
            last_frame_consumed_time = time.time()
        return jsonify({"status": "success", "message": "Generation resumed"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset_generation():
    global active_session_id, last_frame_consumed_time
    try:
        data = request.json or {}
        session_id = data.get("session_id")
        history_length = data.get("history_length")
        smoothing_alpha = data.get("smoothing_alpha")
        if session_id:
            with session_lock:
                if active_session_id and active_session_id != session_id:
                    return jsonify({"status": "error", "message": "Not the active session"}), 403
        if model_manager:
            model_manager.reset(history_length=history_length, smoothing_alpha=smoothing_alpha)
        with session_lock:
            if active_session_id == session_id or not session_id:
                active_session_id = None
        with consumption_monitor_lock:
            last_frame_consumed_time = None
        return jsonify({"status": "success", "message": "Reset complete"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/get_frame", methods=["GET"])
def get_frame():
    global last_frame_consumed_time
    try:
        session_id = request.args.get("session_id")
        if not session_id:
            return jsonify({"status": "error", "message": "session_id is required"}), 400
        if model_manager is None:
            return jsonify({"status": "error", "message": "Model not initialized"}), 400

        count = min(int(request.args.get("count", 8)), 20)
        with session_lock:
            is_active = active_session_id == session_id

        if is_active:
            frames = []
            for _ in range(count):
                joints = model_manager.get_next_frame()
                if joints is None:
                    break
                frames.append(joints.tolist())
            if frames:
                with consumption_monitor_lock:
                    last_frame_consumed_time = time.time()
                return jsonify({
                    "status": "success",
                    "frames": frames,
                    "buffer_size": model_manager.frame_buffer.size(),
                })
        else:
            after_id = int(request.args.get("after_id", 0))
            broadcast = model_manager.get_broadcast_frames(after_id, count)
            if broadcast:
                last_id = broadcast[-1][0]
                frames = [j.tolist() for _, j in broadcast]
                return jsonify({
                    "status": "success",
                    "frames": frames,
                    "last_id": last_id,
                    "buffer_size": model_manager.frame_buffer.size(),
                })

        return jsonify({
            "status": "waiting",
            "message": "No frame available yet",
            "buffer_size": model_manager.frame_buffer.size() if model_manager else 0,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def get_status():
    try:
        session_id = request.args.get("session_id")
        with session_lock:
            is_active_session = bool(session_id and active_session_id == session_id)
            current_active = active_session_id
        if model_manager is None:
            return jsonify({
                "initialized": False,
                "buffer_size": 0,
                "is_generating": False,
                "is_active_session": is_active_session,
                "active_session_id": current_active,
            })
        status = model_manager.get_buffer_status()
        status["initialized"] = True
        status["is_active_session"] = is_active_session
        status["active_session_id"] = current_active
        return jsonify(status)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/export_performance", methods=["POST"])
def export_performance():
    try:
        if model_manager is None:
            return jsonify({"status": "error", "message": "Model not initialized"}), 400
        path = model_manager.export_performance(out_dir=PERF_OUT_DIR)
        print(f"[BAMM] perf exported to {path}")
        return jsonify({"status": "success", "path": path})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BAMM Flask UI server")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--profile", action="store_true",
                        help="Enable performance telemetry. Without it, "
                             "/api/export_performance still works but the "
                             "log message is suppressed.")
    parser.add_argument("--perf-dir", type=str, default=".",
                        help="Where to write the perf JSON file (default: cwd)")
    args = parser.parse_args()
    globals()["PERF_OUT_DIR"] = args.perf_dir
    if args.profile:
        print("[BAMM] --profile enabled; perf telemetry will be exported on demand")
    print("[BAMM] loading model on startup...")
    init_model()
    print(f"[BAMM] starting Flask server on http://127.0.0.1:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
