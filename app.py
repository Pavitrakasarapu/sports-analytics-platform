from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os
import yt_dlp
import threading
import time
import math
from datetime import datetime

from werkzeug.utils import secure_filename
from sports.registry import registry as sport_registry

# Custom modules
from utils.video_processor import VideoProcessor
from utils.player_tracker import PlayerTracker
from utils.ball_tracker import BallTracker
from utils.tactical_analyzer import TacticalAnalyzer
from utils.performance_analyzer import PerformanceAnalyzer
from utils.referee_assistant import RefereeAssistant
from utils.advanced_analytics import AdvancedAnalytics


app = Flask(__name__)
app.config["SECRET_KEY"] = "sports-analytics-secret-key"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB
app.config["ALLOWED_VIDEO_EXTENSIONS"] = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"
}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Flask-SocketIO uses the threading backend in this project.
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
)
CORS(app)


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

active_sessions = {}
video_processors = {}
state_lock = threading.RLock()

# Analytics modules
player_tracker = PlayerTracker()
ball_tracker = BallTracker()
tactical_analyzer = TacticalAnalyzer()
performance_analyzer = PerformanceAnalyzer()
referee_assistant = RefereeAssistant()
advanced_analytics = AdvancedAnalytics()


def get_sport_profile(value):
    """Return a registered sport profile; reject unknown sports cleanly."""
    return sport_registry.get(str(value or "soccer").strip().lower())


def get_session_sport(session_id):
    with state_lock:
        info = active_sessions.get(session_id, {})
        if info.get("sport"):
            return info["sport"]
        processor = video_processors.get(session_id)
        return getattr(processor, "sport", "soccer") if processor else "soccer"


def feature_enabled(sport_key, feature_name):
    profile = get_sport_profile(sport_key)
    if not profile:
        return False
    feature = profile.analytics.get(feature_name)
    return bool(feature and str(getattr(feature.status, "value", feature.status)).lower() != "not supported")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_safe(value):
    """Recursively convert NumPy/scikit-learn values to JSON-safe Python types.

    Advanced analytics uses NumPy/scikit-learn and may return values such as
    np.int32 dictionary keys, np.float32 scalars, ndarrays, tuples, or sets.
    Flask-SocketIO/JSON cannot serialize those objects directly.
    """
    if value is None or isinstance(value, (str, bool)):
        return value

    # Python numeric types and NumPy numeric scalars.
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None

    # NumPy scalar / scalar-like objects (np.int32, np.float32, np.bool_, etc.).
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except Exception:
            pass

    # NumPy arrays.
    tolist_method = getattr(value, "tolist", None)
    if callable(tolist_method):
        try:
            return _json_safe(tolist_method())
        except Exception:
            pass

    if isinstance(value, dict):
        safe_dict = {}
        for key, item in value.items():
            # JSON object keys must be primitive. Stringifying is safest and
            # also handles np.int32/np.int64 keys from clustering labels.
            safe_key = str(_json_safe(key))
            safe_dict[safe_key] = _json_safe(item)
        return safe_dict

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    # Last resort: make the value printable rather than breaking the Socket.IO
    # response.
    return str(value)


def _extract_tracking_frames(tracking_data):
    """Return frames from the supported tracking-data shapes."""
    if isinstance(tracking_data, dict):
        frames = tracking_data.get("frames", [])
    elif isinstance(tracking_data, (list, tuple)):
        frames = tracking_data
    else:
        frames = []

    if isinstance(frames, dict):
        frames = list(frames.values())

    return list(frames) if isinstance(frames, (list, tuple)) else []


def _player_identity(player, frame_index, player_index):
    """Return a stable, JSON-safe identity for a tracked player."""
    if not isinstance(player, dict):
        return f"frame-player:{frame_index}:{player_index}"

    pid = (
        player.get("id")
        if player.get("id") is not None
        else player.get("player_id")
    )
    if pid is None:
        pid = player.get("track_id")
    if pid is None:
        pid = player.get("tracker_id")

    if pid is not None:
        return str(_json_safe(pid))

    center = player.get("center") or player.get("position")
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        try:
            return f"position:{float(center[0]):.2f},{float(center[1]):.2f}"
        except (TypeError, ValueError):
            pass

    bbox = player.get("bbox") or player.get("box")
    if bbox is not None:
        return f"bbox:{_json_safe(bbox)}"

    return f"frame-player:{frame_index}:{player_index}"


def get_processor(session_id):
    """Return a processor safely, or None if the session is unavailable."""
    if not session_id:
        return None
    with state_lock:
        return video_processors.get(session_id)


def emit_error(message):
    """Send a consistent error object to the requesting Socket.IO client."""
    emit("error", {"message": str(message)})


def run_processor(processor, method_name, sid):
    """Run processor work in a daemon thread and report completion/errors."""
    try:
        print(
            f"PROCESSOR START: {method_name} | {processor.session_id}",
            flush=True,
        )

        getattr(processor, method_name)(sid)

        # Uploaded videos are finite. Tell the browser explicitly when the
        # processing loop has returned so it can enable the analytics buttons.
        if method_name == "start_upload_processing":
            with state_lock:
                if processor.session_id in active_sessions:
                    active_sessions[processor.session_id]["running"] = False

            socketio.emit(
                "analysis_completed",
                {
                    "session_id": processor.session_id,
                    "sport": getattr(processor, "sport", "soccer"),
                    "status": "completed",
                    "message": "Video processing completed. You can now run all analyses.",
                },
                to=sid,
            )

        print(
            f"PROCESSOR END: {method_name} | {processor.session_id}",
            flush=True,
        )

    except Exception as exc:
        print(
            f"Processing error for session {processor.session_id}: {exc}",
            flush=True,
        )
        import traceback
        traceback.print_exc()

        socketio.emit(
            "error",
            {
                "message": f"Processing failed: {exc}",
                "session_id": processor.session_id,
            },
            to=sid,
        )
        # Keep the processor so the frontend can request whatever tracking
        # data was collected before the error.


def cleanup_session(session_id):
    """Stop a session but retain processor/tracking data for post-analysis."""
    with state_lock:
        processor = video_processors.get(session_id)
        if session_id in active_sessions:
            active_sessions[session_id]["running"] = False

    if processor is not None:
        try:
            processor.stop()
        except Exception as exc:
            print(f"Session cleanup warning ({session_id}): {exc}", flush=True)


# ---------------------------------------------------------------------------
# Phase 2 - Multi-sport API
# ---------------------------------------------------------------------------

@app.route("/api/sports")
def list_sports():
    return jsonify({"sports": sport_registry.as_dict()})


@app.route("/api/sports/<sport_key>")
def get_sport(sport_key):
    profile = get_sport_profile(sport_key)
    if not profile:
        return jsonify({"error": f"Unsupported sport: {sport_key}"}), 400
    return jsonify(profile.to_dict())


@socketio.on("get_sport_capabilities")
def handle_sport_capabilities(data=None):
    profile = get_sport_profile((data or {}).get("sport", "soccer"))
    if not profile:
        emit_error(f"Unsupported sport: {(data or {}).get('sport')}")
        return
    emit("sport_capabilities", profile.to_dict())


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Main application page."""
    return render_template("index.html")


@app.route("/api/health")
def health_check():
    """Health check endpoint."""
    with state_lock:
        session_count = len(active_sessions)

    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "active_sessions": session_count,
        }
    )

@app.route("/api/youtube", methods=["POST"])
def youtube_video():
    """Download a YouTube video and create an analysis session."""
    data = request.get_json(silent=True) or {}

    url = data.get("url", "").strip()
    sport = data.get("sport", "soccer")

    if not url:
        return jsonify({"error": "YouTube URL is required"}), 400

    profile = get_sport_profile(sport)
    if not profile:
        return jsonify({"error": f"Unsupported sport: {sport}"}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_id = f"youtube_{timestamp}"

    output_template = os.path.join(
        app.config["UPLOAD_FOLDER"],
        f"{timestamp}_youtube.%(ext)s"
    )

    try:
        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "merge_output_format": "mp4",
            "js_runtimes": {"deno": {}},
            "outtmpl": output_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_file = ydl.prepare_filename(info)

        if not os.path.exists(downloaded_file):
            possible_files = [
                os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    f
                )
                for f in os.listdir(app.config["UPLOAD_FOLDER"])
                if f.startswith(f"{timestamp}_youtube.")
            ]

            if possible_files:
                downloaded_file = possible_files[0]

        if not os.path.exists(downloaded_file):
            raise ValueError("YouTube video could not be downloaded")

        processor = VideoProcessor(
            video_path=downloaded_file,
            session_id=session_id,
            socketio=socketio,
            sport=profile.key,
        )

        with state_lock:
            video_processors[session_id] = processor
            active_sessions[session_id] = {
                "type": "youtube",
                "sport": profile.key,
                "running": False,
            }

        filename = os.path.basename(downloaded_file)

        print(
            f"YOUTUBE SUCCESS: {url} -> {downloaded_file} -> {session_id}",
            flush=True,
        )

        return jsonify({
            "session_id": session_id,
            "filename": filename,
            "original_filename": info.get("title", "YouTube Video"),
            "size": os.path.getsize(downloaded_file),
            "video_url": f"/uploads/{filename}",
            "sport": profile.key,
            "sport_name": profile.name,
            "capabilities": profile.to_dict(),
            "message": "YouTube video downloaded successfully. Ready for analysis."
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()

        return jsonify({"error": str(exc)}), 500
@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Upload a video, create its analysis session, and return the session id."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    sport = request.form.get("sport", "soccer")
    profile = get_sport_profile(sport)
    if not profile:
        return jsonify({"error": f"Unsupported sport: {sport}"}), 400

    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400

    original_name = secure_filename(file.filename)
    if not original_name:
        return jsonify({"error": "Invalid filename"}), 400

    extension = os.path.splitext(original_name)[1].lower()
    if extension not in app.config["ALLOWED_VIDEO_EXTENSIONS"]:
        allowed = ", ".join(sorted(app.config["ALLOWED_VIDEO_EXTENSIONS"]))
        return jsonify(
            {"error": f"Unsupported video format. Allowed: {allowed}"}
        ), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{original_name}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(filepath)

        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            raise ValueError("Uploaded file is empty or could not be saved")

        session_id = f"upload_{timestamp}"

        processor = VideoProcessor(
            video_path=filepath,
            session_id=session_id,
            socketio=socketio,
            sport=profile.key,
        )

        with state_lock:
            video_processors[session_id] = processor
            active_sessions[session_id] = {
                "type": "upload",
                "sport": profile.key,
                "running": False,
            }

        print(
            f"UPLOAD SUCCESS: {original_name} -> {filepath} -> {session_id}",
            flush=True,
        )

        return jsonify(
            {
                "session_id": session_id,
                "filename": filename,
                "original_filename": original_name,
                "size": os.path.getsize(filepath),
                "video_url": f"/uploads/{filename}",
                "sport": profile.key,
                "sport_name": profile.name,
                "capabilities": profile.to_dict(),
                "message": "Video uploaded successfully. Ready for analysis.",
            }
        )

    except Exception as exc:
        import traceback
        traceback.print_exc()

        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except OSError:
            pass

        return jsonify({"error": str(exc)}), 500


@app.route("/uploads/<path:filename>")
def uploaded_video_file(filename):
    """Serve uploaded videos for browser preview."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/api/analysis/<session_id>")
def get_analysis(session_id):
    """Get analysis results for a session."""
    processor = get_processor(session_id)

    if processor is None:
        return jsonify({"error": "Session not found"}), 404

    try:
        return jsonify(processor.get_analysis_results())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/sessions")
def list_sessions():
    """List all currently stored sessions."""
    with state_lock:
        processors = list(video_processors.items())

    sessions = []
    for session_id, processor in processors:
        try:
            status = processor.get_status()
        except Exception:
            status = "unknown"

        created_at = getattr(processor, "created_at", None)
        sessions.append(
            {
                "session_id": session_id,
                "status": status,
                "created_at": (
                    created_at.isoformat()
                    if hasattr(created_at, "isoformat")
                    else None
                ),
            }
        )

    return jsonify(sessions)


# ---------------------------------------------------------------------------
# Socket.IO connection
# ---------------------------------------------------------------------------

@socketio.on("connect")
def handle_connect():
    print(f"Client connected: {request.sid}", flush=True)
    emit("connected", {"status": "connected"})


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    print(f"Client disconnected: {sid}", flush=True)

    # Copy IDs first so cleanup can safely mutate the dictionaries.
    with state_lock:
        owned_sessions = [
            session_id
            for session_id, info in active_sessions.items()
            if info.get("client_id") == sid
        ]

    for session_id in owned_sessions:
        with state_lock:
            if session_id in active_sessions:
                active_sessions[session_id]["client_id"] = None


# ---------------------------------------------------------------------------
# Live analysis
# ---------------------------------------------------------------------------

@socketio.on("start_live_analysis")
def handle_start_live_analysis(data=None):
    """Start live camera analysis."""
    try:
        data = data or {}
        camera_index = data.get("camera_index", 0)
        profile = get_sport_profile(data.get("sport", "soccer"))
        if not profile:
            emit_error(f"Unsupported sport: {data.get('sport')}")
            return

        try:
            camera_index = int(camera_index)
        except (TypeError, ValueError):
            camera_index = 0

        session_id = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        processor = VideoProcessor(
            camera_index=camera_index,
            session_id=session_id,
            socketio=socketio,
            sport=profile.key,
        )

        with state_lock:
            video_processors[session_id] = processor
            active_sessions[session_id] = {
                "client_id": request.sid,
                "type": "live",
                "camera_index": camera_index,
                "sport": profile.key,
            }

        processor.sport = profile.key
        processor.tracking_data["sport"] = profile.key

        thread = threading.Thread(
            target=run_processor,
            args=(processor, "start_live_processing", request.sid),
            daemon=True,
        )
        thread.start()

        print(f"Started live processing for session {session_id}", flush=True)

        emit(
            "analysis_started",
            {
                "session_id": session_id,
                "status": "started",
                "sport": profile.key,
                "sport_name": profile.name,
                "capabilities": profile.to_dict(),
            },
        )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        emit_error(exc)


# ---------------------------------------------------------------------------
# Uploaded-video analysis
# ---------------------------------------------------------------------------

@socketio.on("start_upload_analysis")
def handle_start_upload_analysis(data=None):
    """Start processing for an already-uploaded video session."""
    try:
        data = data or {}
        session_id = data.get("session_id")

        if not session_id:
            emit_error("Session ID is required")
            return

        processor = get_processor(session_id)

        if processor is None:
            emit_error("Upload session not found")
            return

        sport = get_session_sport(session_id)
        profile = get_sport_profile(data.get("sport") or sport)
        if not profile:
            emit_error("Unsupported sport")
            return

        with state_lock:
            existing = active_sessions.get(session_id, {})
            existing["client_id"] = request.sid
            existing["sport"] = profile.key

            # Prevent accidental double-starts from repeated clicks.
            if existing and existing.get("running"):
                emit(
                    "analysis_started",
                    {
                        "session_id": session_id,
                        "status": "already_running",
                    },
                )
                return

            active_sessions[session_id] = {
                **existing,
                "client_id": request.sid,
                "type": "upload",
                "sport": profile.key,
                "running": True,
            }

        thread = threading.Thread(
            target=run_processor,
            args=(processor, "start_upload_processing", request.sid),
            daemon=True,
        )
        thread.start()

        print(
            f"Started uploaded-video processing for session {session_id}",
            flush=True,
        )

        emit(
            "analysis_started",
            {
                "session_id": session_id,
                "status": "started",
                "sport": profile.key,
                "sport_name": profile.name,
                "capabilities": profile.to_dict(),
            },
        )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        emit_error(exc)


# ---------------------------------------------------------------------------
# Stop analysis
# ---------------------------------------------------------------------------

@socketio.on("stop_analysis")
def handle_stop_analysis(data=None):
    """Stop analysis and release the session."""
    data = data or {}
    session_id = data.get("session_id")

    if not session_id:
        emit_error("Session ID is required")
        return

    processor = get_processor(session_id)
    if processor is None:
        emit_error("Session not found")
        return

    cleanup_session(session_id)

    print(f"Stopped processing for session {session_id}", flush=True)

    emit(
        "analysis_stopped",
        {
            "session_id": session_id,
            "status": "stopped",
        },
    )


# ---------------------------------------------------------------------------
# Tactical insights
# ---------------------------------------------------------------------------

@socketio.on("get_tactical_insights")
def handle_tactical_insights(data=None):
    """Generate and return tactical insights for a live/upload session."""
    print(f"TACTICAL EVENT RECEIVED: {data}", flush=True)

    try:
        data = data or {}
        session_id = data.get("session_id")
        print(f"TACTICAL SESSION ID: {session_id}", flush=True)

        processor = get_processor(session_id)

        if processor is None:
            emit_error("Session not found")
            return

        tracking_data = processor.get_tracking_data()

        if not tracking_data:
            emit(
                "tactical_insights",
                {
                    "formation": None,
                    "spacing": None,
                    "passing": None,
                    "possession": None,
                    "pressing": None,
                    "transition": None,
                    "set_piece": None,
                    "heatmap": None,
                    "message": "No tracking data is available yet.",
                },
            )
            return

        sport = get_session_sport(session_id)
        profile = get_sport_profile(sport)
        print(f"CALLING TACTICAL ANALYZER | sport={sport}", flush=True)

        if sport == "soccer":
            insights = tactical_analyzer.get_tactical_insights(tracking_data)
        else:
            # Reuse only sport-safe movement visualizations. Do not expose
            # football formation/offside/possession semantics to other sports.
            frames = _extract_tracking_frames(tracking_data)
            latest = frames[-1] if frames else {}
            players = latest.get("players", []) if isinstance(latest, dict) else []
            insights = {
                "formation": None,
                "spacing": tactical_analyzer.analyze_player_spacing(players, 1920, 1080) if len(players) >= 2 else None,
                "heatmap": tactical_analyzer.generate_heatmap(players, 1920, 1080) if players else None,
                "passing": None, "possession": None, "pressing": None,
                "transition": None, "set_piece": None,
                "sport": get_session_sport(session_id),
                "sport_name": profile.name if profile else sport,
                "message": "Generic movement analysis only. Sport-specific tactical events are not enabled for this sport yet."
            }

        print("TACTICAL ANALYZER COMPLETED", flush=True)

        if insights is None:
            insights = {
                "formation": None,
                "spacing": None,
                "passing": None,
                "possession": None,
                "pressing": None,
                "transition": None,
                "set_piece": None,
                "heatmap": None,
                "message": "Not enough tracking data for tactical analysis.",
            }

        insights["sport"] = sport
        insights["sport_name"] = profile.name if profile else sport
        emit("tactical_insights", _json_safe(insights))

    except Exception as exc:
        import traceback
        traceback.print_exc()
        emit_error(exc)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

@socketio.on("get_performance_metrics")
def handle_performance_metrics(data=None):
    """Generate and return performance metrics."""
    try:
        data = data or {}
        session_id = data.get("session_id")
        processor = get_processor(session_id)

        if processor is None:
            emit_error("Session not found")
            return

        tracking_data = processor.get_tracking_data()

        if not tracking_data:
            emit(
                "performance_metrics",
                {
                    "message": "No tracking data is available yet.",
                },
            )
            return

        metrics = performance_analyzer.get_performance_metrics(tracking_data)
        if metrics is None:
            metrics = {"individual_metrics": {}, "team_metrics": {}, "team_coordination": {}}
        metrics["sport"] = get_session_sport(session_id)
        emit("performance_metrics", _json_safe(metrics))

    except Exception as exc:
        import traceback
        traceback.print_exc()
        emit_error(exc)


# ---------------------------------------------------------------------------
# Advanced analytics
# ---------------------------------------------------------------------------

@socketio.on("get_advanced_analytics")
def handle_advanced_analytics(data=None):
    """Generate advanced analytics and always return JSON-safe results."""
    try:
        data = data or {}
        session_id = data.get("session_id")
        processor = get_processor(session_id)

        if processor is None:
            emit_error("Session not found")
            return
        sport = get_session_sport(session_id)
        profile = get_sport_profile(sport)

        tracking_data = processor.get_tracking_data() or {}
        frames = _extract_tracking_frames(tracking_data)

        if not frames:
            payload = {
                "clustering": None,
                "anomalies": None,
                "team_synchronization": None,
                "heatmap_visualization": None,
                "summary": {
                    "frames": 0,
                    "players_detected": 0,
                    "player_observations": 0,
                },
                "errors": [],
                "message": "No tracking data is available yet.",
                "timestamp": time.time(),
            }
            emit("advanced_analytics", _json_safe(payload))
            return

        results = {
            "clustering": None,
            "anomalies": None,
            "team_synchronization": None,
            "heatmap_visualization": None,
        }
        errors = []

        # Run each advanced analysis independently. One analyzer failing must
        # not prevent the other results from reaching the UI.
        analyzers = [
            ("clustering", advanced_analytics.analyze_player_clustering),
            ("anomalies", advanced_analytics.detect_anomalies),
            (
                "team_synchronization",
                advanced_analytics.analyze_team_synchronization,
            ),
            (
                "heatmap_visualization",
                advanced_analytics.generate_heatmap_visualization,
            ),
        ]

        for key, analyzer in analyzers:
            try:
                results[key] = analyzer(tracking_data)
            except Exception as exc:
                print(f"ADVANCED {key.upper()} ERROR: {exc}", flush=True)
                import traceback
                traceback.print_exc()
                errors.append(f"{key}: {exc}")

        player_ids = set()
        total_player_observations = 0

        for frame_index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue

            players = frame.get("players")
            if players is None:
                players = frame.get("player_positions")
            if players is None:
                players = []

            if isinstance(players, dict):
                players = list(players.values())
            elif not isinstance(players, (list, tuple)):
                players = []

            total_player_observations += len(players)

            for player_index, player in enumerate(players):
                player_ids.add(
                    _player_identity(player, frame_index, player_index)
                )

        payload = {
            **results,
            "sport": get_session_sport(session_id),
            "sport_name": profile.name if profile else sport,
            "summary": {
                "frames": len(frames),
                "players_detected": len(player_ids),
                "player_observations": total_player_observations,
            },
            "errors": errors,
            "message": (
                "Advanced analytics completed."
                if not errors
                else "Advanced analytics completed with limited results."
            ),
            "timestamp": time.time(),
        }

        # CRITICAL FIX:
        # scikit-learn/NumPy can return np.int32 keys and np.float32 values.
        # Flask-SocketIO's JSON encoder cannot serialize those dictionary keys.
        payload = _json_safe(payload)

        print("ADVANCED ANALYTICS RESULT:", payload, flush=True)
        emit("advanced_analytics", payload)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        emit_error(exc)


# ---------------------------------------------------------------------------
# Prediction analysis
# ---------------------------------------------------------------------------

@socketio.on("get_prediction_analysis")
@socketio.on("get_predictions")
def handle_prediction_analysis(data=None):
    """Generate safe performance trend predictions."""
    try:
        data = data or {}
        session_id = data.get("session_id")
        processor = get_processor(session_id)

        if processor is None:
            emit_error("Session not found")
            return

        tracking_data = processor.get_tracking_data() or []

        if not tracking_data:
            emit(
                "prediction_analysis",
                {
                    "trend_analysis": {
                        "work_rate_trend": "Stable",
                        "intensity_trend": "Stable",
                        "fatigue_trend": "Stable",
                    },
                    "predictions": [],
                    "prediction_horizon": 1,
                    "message": "No tracking data is available yet.",
                    "timestamp": time.time(),
                },
            )
            return

        # Performance metrics already work in the current application.
        metrics = performance_analyzer.get_performance_metrics(
            tracking_data
        ) or {}

        if not isinstance(metrics, dict):
            metrics = {}

        team_metrics = metrics.get("team_metrics") or {}
        if not isinstance(team_metrics, dict):
            team_metrics = {}

        def safe_float(value, default=0.0):
            if value is None:
                return default
            try:
                number = float(value)
                return number if number == number else default
            except (TypeError, ValueError):
                return default

        work_rate = safe_float(
            team_metrics.get("team_work_rate"),
            0.0,
        )
        average_speed = safe_float(
            team_metrics.get("average_speed"),
            0.0,
        )
        total_sprints = safe_float(
            team_metrics.get("total_sprints"),
            0.0,
        )

        # Keep values bounded and deterministic. This avoids the previous
        # float(None) failure inside the prediction path.
        predicted_intensity = min(
            max(average_speed * 10.0, 0.0),
            100.0,
        )

        predicted_fatigue = min(
            max(
                (total_sprints * 0.05)
                + (work_rate / 10000.0),
                0.0,
            ),
            1.0,
        )

        if work_rate > 1000:
            work_rate_trend = "High"
        elif work_rate > 300:
            work_rate_trend = "Moderate"
        else:
            work_rate_trend = "Stable"

        if predicted_intensity > 70:
            intensity_trend = "High"
        elif predicted_intensity > 30:
            intensity_trend = "Moderate"
        else:
            intensity_trend = "Low"

        if predicted_fatigue > 0.7:
            fatigue_trend = "High"
        elif predicted_fatigue > 0.3:
            fatigue_trend = "Moderate"
        else:
            fatigue_trend = "Low"

        result = {
            "sport": get_session_sport(session_id),
            "trend_analysis": {
                "work_rate_trend": work_rate_trend,
                "intensity_trend": intensity_trend,
                "fatigue_trend": fatigue_trend,
            },
            "predictions": [
                {
                    "predicted_work_rate": round(work_rate, 2),
                    "predicted_intensity": round(predicted_intensity, 2),
                    "predicted_fatigue": round(predicted_fatigue, 3),
                }
            ],
            "prediction_horizon": 1,
            "current_metrics": {
                "work_rate": round(work_rate, 2),
                "average_speed": round(average_speed, 2),
                "total_sprints": round(total_sprints, 2),
            },
            "timestamp": time.time(),
        }

        print("PREDICTION RESULT:", result, flush=True)
        emit("prediction_analysis", result)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        emit_error(exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting Sports Analytics Application...")
    print("Backend server will be available at http://localhost:5000")
    print("Frontend should be running at http://localhost:3000")

    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=True,
    )
