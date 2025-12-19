#!/usr/bin/env -S python3 -O
"""
Flask app:
 - Serves web.html
 - Acts as a PROXY: Fetches data from Robot WiFi
 - CALCULATES HAZARD RISKS using XGBoost (with threshold fallback)
 - Logs to MySQL ONLY when Recording is Active
"""

import os
import sys

# FIX PATH FIRST - before any other imports that need grid_state
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    print(f"✅ Added to sys.path: {parent_dir}")

# Now import grid_state (after path is set)
import grid_state

# Test grid_state import
try:
    print("✅ grid_state imported successfully")
    
    # Test write/read
    test_data = b"test123"
    grid_state.set_grid(test_data)
    read_data = grid_state.get_grid()
    
    if read_data == test_data:
        print("✅ grid_state read/write working!")
    else:
        print("❌ grid_state read/write failed")
        
except Exception as e:
    print(f"❌ grid_state import/test failed: {e}")
    import traceback
    traceback.print_exc()

import threading
import time
import requests
import mysql.connector
import math 
import base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from upload import save_uploaded_file 

from hazard_nlp_classifier import get_classifier
hazard_classifier = get_classifier(
    model_path='models/nlp/logistic.pkl',
    model_type='logistic'
)

# Debug: Check if model loaded
print(f"\n{'='*80}")
print(f"MODEL LOAD STATUS CHECK:")
print(f"  Model object: {hazard_classifier.model}")
print(f"  Model type: {hazard_classifier.model_type}")
print(f"  Tokenizer: {hazard_classifier.tokenizer}")
print(f"  Model path exists: {hazard_classifier.model_path.exists()}")
print(f"{'='*80}\n")

# ============================================================================
# MODEL CONFIGURATION - CHANGE THIS TO TEST DIFFERENT MODELS
# ============================================================================

# Available models and their configurations
AVAILABLE_MODELS = {
    'logistic': {
        'path': 'models/nlp/logistic_regression.pkl',
        'type': 'logistic',
        'name': 'Logistic Regression (BoW)',
        'description': 'Fast, interpretable baseline'
    },
    'naive_bayes': {
        'path': 'models/nlp/naive_bayes.pkl',
        'type': 'naive_bayes',
        'name': 'Naive Bayes (TF-IDF)',
        'description': 'Probabilistic baseline'
    },
    'decision_tree': {
        'path': 'models/nlp/decision_tree.pkl',
        'type': 'decision_tree',
        'name': 'Decision Tree',
        'description': 'Rule-based classifier'
    },
    'random_forest': {
        'path': 'models/nlp/random_forest.pkl',
        'type': 'random_forest',
        'name': 'Random Forest',
        'description': 'Ensemble method'
    },
    'lstm': {
        'path': 'models/nlp/lstm_model.h5',
        'type': 'lstm',
        'name': 'LSTM',
        'description': 'Recurrent neural network'
    },
    'cnn': {
        'path': 'models/nlp/cnn_model.h5',
        'type': 'cnn',
        'name': 'CNN (Kim 2014)',
        'description': 'Convolutional text model'
    },
    'transformer': {
        'path': 'models/nlp/transformer_model.h5',
        'type': 'transformer',
        'name': 'Tiny Transformer',
        'description': 'Attention-based model'
    },
    'xgboost': {
        'path': 'models/hazard_xgboost_model.json',
        'type': 'xgboost',
        'name': 'XGBoost (Original)',
        'description': 'Your original XGBoost model'
    }
}

# ============================================================================
# CHOOSE YOUR MODEL HERE - Change this line to test different models
# ============================================================================
ACTIVE_MODEL = 'cnn'  # Options: 'logistic', 'naive_bayes', 'cnn', 'transformer', etc.
# ============================================================================

# Validate model choice
if ACTIVE_MODEL not in AVAILABLE_MODELS:
    print(f"❌ Invalid model choice: {ACTIVE_MODEL}")
    print(f"Available models: {', '.join(AVAILABLE_MODELS.keys())}")
    sys.exit(1)

model_config = AVAILABLE_MODELS[ACTIVE_MODEL]

# Load the selected model
print(f"\n{'='*80}")
print(f"  LOADING MODEL: {model_config['name']}")
print(f"  Description: {model_config['description']}")
print(f"  Path: {model_config['path']}")
print(f"{'='*80}\n")

# Initialize classifier
if ACTIVE_MODEL == 'xgboost':
    # Use original XGBoost classifier
    from hazard_xgboost_model import get_classifier as get_xgb_classifier
    hazard_classifier = get_xgb_classifier()
else:
    # Use NLP classifier
    hazard_classifier = get_classifier(
        model_path=model_config['path'],
        model_type=model_config['type']
    )
# ============================================================================

# CONFIGURATION
MYSQL_CONFIG = {
    'host': '127.0.0.1',  
    'user': 'root',  
    'password': 'Bangtan_0613',
    'database': 'bimbot_db'      
}

ROBOT_SENSOR_URL = "http://192.168.4.1/dht" 
UPLOAD_DIR = Path("data/videos/uploads") 
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# --- GLOBAL STATE ---


latest_sensor_data = {
    "temperature": 0.0,
    "humidity": 0.0,
    "heat_index": 0.0,
    "risk_level": "Initializing...",
    "risk_score": 0,
    "confidence": 0.0,
    "probabilities": {"low": 0.0, "moderate": 0.0, "severe": 0.0},  # <-- add
    "method": "initializing",
    "timestamp": "Waiting for Robot...",
    "is_recording": False,
    "interpretation": "--",
}

latest_occupancy_grid = None

def set_occupancy_grid(grid_bytes):
    """Called by SLAM thread to update grid"""
    global latest_occupancy_grid
    latest_occupancy_grid = grid_bytes

recording_active = False 

# --- DATABASE HELPER ---
def store_data_in_db(temp, hum, hi, risk_label, risk_score, confidence, method, current_time):
    """Store sensor data and ML predictions in database"""
    try:
        db = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = db.cursor()
        
        # Updated schema to include confidence and method
        sql = """
            INSERT INTO sensor_data 
            (temperature, humidity, heat_index, risk_level, risk_score, 
             confidence, classification_method, timestamp) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        val = (temp, hum, hi, risk_label, risk_score, confidence, method, current_time)
        cursor.execute(sql, val)
        db.commit()
        
        method_icon = "🤖" if method == "xgboost" else "📊"
        print(f"💾 {method_icon} Saved to MySQL (conf={confidence:.2f})")
    except mysql.connector.Error as err:
        print(f"DB ERROR: {err}")
    finally:
        if 'db' in locals() and db.is_connected():
            cursor.close()
            db.close()

# --- WIFI SENSOR LISTENER ---
def wifi_sensor_listener():
    global latest_sensor_data, recording_active
    print(f"📡 Listener started. Polling {ROBOT_SENSOR_URL}...")

    while True:
        try:
            response = requests.get(ROBOT_SENSOR_URL, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                
                temp = float(data.get('temp_c', 0))
                hum = float(data.get('humidity', 0))
                current_time_str = time.strftime("%Y-%m-%d %H:%M:%S")

                # === USE PRE-TRAINED MODEL FOR HAZARD CLASSIFICATION ===
                risk_label, risk_score, confidence, prob_dict, method, interpretation = hazard_classifier.predict(temp, hum)

                # Normalize probabilities to a named object
                if isinstance(prob_dict, dict):
                    probs_obj = {
                        "low": round(float(prob_dict.get("low", 0.0)), 3),
                        "moderate": round(float(prob_dict.get("moderate", 0.0)), 3),
                        "severe": round(float(prob_dict.get("severe", 0.0)), 3),
                    }
                else:
                    probs = list(prob_dict) if prob_dict is not None else []
                    probs_obj = {
                        "low": round(float(probs[0]), 3) if len(probs) > 0 else 0.0,
                        "moderate": round(float(probs[1]), 3) if len(probs) > 1 else 0.0,
                        "severe": round(float(probs[2]), 3) if len(probs) > 2 else 0.0,
                    }

                # Save to DB if recording
                if recording_active:
                    store_data_in_db(
                        temp, hum,
                        calculate_heat_index(temp, hum),
                        risk_label, risk_score,
                        confidence, method,
                        current_time_str
                    )

                # Update Global State for API
                latest_sensor_data = {
                    "temperature": round(temp, 2),
                    "humidity": round(hum, 2),
                    "heat_index": round(calculate_heat_index(temp, hum), 2),
                    "risk_level": risk_label,
                    "risk_score": risk_score,
                    "confidence": round(confidence, 3),
                    "probabilities": probs_obj,      # <-- use normalized object
                    "method": method,
                    "timestamp": current_time_str,
                    "is_recording": recording_active,
                    "interpretation": interpretation,
                }

                # Enhanced status logging
                status_icon = "🔴 REC" if recording_active else "⚪ IDLE"
                method_icon = "🤖 XGB" if method == "xgboost" else "📊 THR"
                print(f"{status_icon} | {method_icon} | {temp}°C | {hum}% | "
                      f"{risk_label} (conf={confidence:.2f})")

            else:
                print(f"⚠️ Robot Status: {response.status_code}")

        except Exception as e:
            print(f"[wifi_sensor_listener] Error: {e}")

        time.sleep(2)

# Helper function (keep for backward compatibility)
def calculate_heat_index(temp_c, rh):
    """Calculate heat index - kept for database storage"""
    if temp_c is None or rh is None:
        return 0.0
    T = temp_c * 9 / 5 + 32.0
    R = rh
    HI_f = (-42.379 + 2.04901523 * T + 10.14333127 * R - 0.22475541 * T * R
            - 0.00683783 * T ** 2 - 0.05481717 * R ** 2 + 0.00122874 * T ** 2 * R
            + 0.00085282 * T * R ** 2 - 0.00000199 * T ** 2 * R ** 2)
    if R < 13 and 80 <= T <= 112:
        adj = ((13 - R) / 4) * ((17 - abs(T - 95.0)) / 17)**0.5
        HI_f -= adj
    elif R > 85 and 80 <= T <= 87:
        adj = ((R - 85) / 10) * ((87 - T) / 5)
        HI_f += adj
    return (HI_f - 32.0) * 5 / 9

# -----------------------------------------------------------------------------
# Ensure repo root is on sys.path when running from bimbot/
# -----------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from main_slam_dashboard import run_slam  # relies on REPO_ROOT being in sys.path

# --- FLASK APP ---
app = Flask(__name__)
CORS(app) 

@app.route("/", methods=["GET"])
def index():
    return render_template("web.html")

@app.route("/toggle_recording", methods=["POST"])
def toggle_recording():
    global recording_active
    data = request.get_json()
    action = data.get('action')
    
    if action == 'start':
        recording_active = True
        print(">>> STARTING DATABASE RECORDING (XGBoost Mode) <<<")
        return jsonify({"status": "success", "state": "recording"})
    elif action == 'stop':
        recording_active = False
        print(">>> STOPPING DATABASE RECORDING <<<")
        return jsonify({"status": "success", "state": "stopped"})
    
    return jsonify({"status": "error"}), 400

# -----------------------------------------------------------------------------
# Upload configuration
# -----------------------------------------------------------------------------
UPLOAD_DIR = Path("/home/sophia/pyslam/data/videos/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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

@app.route('/api/sensor-data', methods=['GET'])
def get_sensor_data():
    """Return latest sensor data with XGBoost predictions"""
    return jsonify(latest_sensor_data)

@app.route("/health/db", methods=["GET"])
def health_db():
    try:
        db = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = db.cursor()
        cursor.execute("SELECT @@hostname, @@port, CURRENT_USER(), USER(), DATABASE();")
        host, port, current_user, user_string, current_db = cursor.fetchone()
        cursor.execute("SELECT COUNT(*) FROM sensor_data;")
        count = cursor.fetchone()[0]
        cursor.close()
        db.close()
        
        # Check if XGBoost is active
        model_status = "XGBoost Active" if hazard_classifier.model is not None else "Threshold Fallback"
        
        return jsonify({
            "db": "ok",
            "host": host,
            "port": port,
            "current_user": current_user,
            "user_string": user_string,
            "database": current_db,
            "rows": count,
            "ml_model": model_status  # NEW: Model status
        }), 200
    except mysql.connector.Error as err:
        return jsonify({"db": "error", "message": str(err)}), 500
    
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

@app.route('/api/occupancy-grid', methods=['GET'])
def get_occupancy_grid():
    """Return latest occupancy grid as base64 image"""
    try:
        grid_bytes = grid_state.get_grid()
        
        if grid_bytes is None:
            timestamp, size = grid_state.get_metadata()
            if timestamp:
                age = time.time() - timestamp
                print(f"[API] Grid data exists but is {age:.1f}s old ({size} bytes)")
            else:
                print("[API] No grid data available yet")
            return jsonify({"status": "no_data", "image": None})
        
        # Convert to base64
        img_base64 = base64.b64encode(grid_bytes).decode('utf-8')
        
        print(f"[API] Returning grid: {len(grid_bytes)} bytes")
        
        return jsonify({
            "status": "success",
            "image": f"data:image/jpeg;base64,{img_base64}",
            "timestamp": time.time()
        })
        
    except Exception as e:
        print(f"[API ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/stream/occupancy-grid')
def stream_occupancy_grid():
    """Stream occupancy grid as MJPEG"""
    def generate():
        import time
        while True:
            grid_bytes = grid_state.get_grid()
            if grid_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + grid_bytes + b'\r\n')
            time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    wifi_thread = threading.Thread(target=wifi_sensor_listener)
    wifi_thread.daemon = True 
    wifi_thread.start()
    print("🚀 BIM-BOT Server Running on Port 4000...")
    print(f"🤖 ML Model: {'XGBoost' if hazard_classifier.model else 'Threshold Fallback'}")
    app.run(debug=True, host="0.0.0.0", port=4000, use_reloader=False)

    