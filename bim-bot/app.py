#!/usr/bin/env -S python3 -O
"""
Flask app:
 - Serves web.html
 - Accepts video upload, saves to /home/sophia/bim-bot/data/uploads, generates timestamps
 - Invokes SLAM via /home/sophia/pyslam/main_slam.py

Run: ./app.py
"""

import os
import sys
import cv2
import threading

from flask import Flask, render_template, request, jsonify
from upload import save_uploaded_file
from pathlib import Path

# -----------------------------------------------------------------------------
# Ensure repo root is on sys.path when running from bimbot/
# -----------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from main_slam_copy import run_slam  # relies on REPO_ROOT being in sys.path
# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    """
    Render the landing page. Ensure templates/web.html exists relative to bimbot/.
    e.g., bimbot/templates/web.html
    """
    return render_template("web.html")

# -----------------------------------------------------------------------------
# Upload configuration
# -----------------------------------------------------------------------------
UPLOAD_DIR = Path("/home/sophia/pyslam/data/videos/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload():
    """
    Accepts a 'video' file via multipart/form-data and saves it under UPLOAD_DIR.
    Naming policy:
      - New file saved as: new_upload.<ext>
      - If new_upload.<ext> exists, it is renamed to old_upload(n).<ext> (n increments).
    """
    file = request.files.get("video")
    if file is None:
        return jsonify({"status": "error", "error": "No file attached (form field 'video' not found)"}), 400


    ok, info = save_uploaded_file(
        file=file,
        target_dir=UPLOAD_DIR,
        new_base="new_upload",
        old_base="old_upload",
    )

    if not ok:
        return jsonify({"status": "error", "error": info.get("error"), "data": info}), 400

    return jsonify({"status": "partial", "data": info}), 200


@app.route("/generate_map", methods=["POST"])
def generate_map():
    """
    Triggers SLAM by calling run_slam(...) directly (no subprocess).
    Optional JSON body:
        {
          "headless": true,               # default true
          "config_path": "/path/to.yaml", # optional
          "no_output_date": false,        # optional
          "async": true                   # default true -> background thread
        }
    """
    try:
        payload = request.get_json(silent=True) or {}
        headless = False
        config_path = payload.get("config_path")
        no_output_date = payload.get("no_output_date", False)
        run_async = payload.get("async", False)

        if run_async:
            t = threading.Thread(
                target=run_slam,
                kwargs={
                    "headless": headless,
                    "config_path": config_path,
                    "no_output_date": no_output_date,
                },
                daemon=True
            )
            t.start()
            return jsonify({
                "status": "success",
                "message": "Map generation started (async)",
                "params": {
                    "headless": headless,
                    "config_path": config_path,
                    "no_output_date": no_output_date
                }
            }), 202
        else:
            # Synchronous (blocks until SLAM finishes)
            run_slam(
                headless=headless,
                config_path=config_path,
                no_output_date=no_output_date
            )
            return jsonify({
                "status": "success",
                "message": "Map generation finished",
                "params": {
                    "headless": headless,
                    "config_path": config_path,
                    "no_output_date": no_output_date
                }
            }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Bind to 0.0.0.0 to be reachable on LAN; change port if needed
    app.run(debug=True, host="0.0.0.0", port=5000)
