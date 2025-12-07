from flask import Flask, request, jsonify, render_template
from pathlib import Path
import shutil

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("web.html")

UPLOAD_DIR = Path(r"D:\Sophia\Videos\bimbot_loaded_videos")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return "No file attached", 400

    file = request.files["video"]

    if file.filename == "":
        return "Empty filename", 400

    dest = UPLOAD_DIR / file.filename
    file.save(dest)

    return "OK", 200

if __name__ == "__main__":
    app.run(debug=True)
