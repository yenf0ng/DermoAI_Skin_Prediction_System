from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tempfile
from pathlib import Path
import sys

# =========================================================
# PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "image_pipeline"))

from integrate_model import predict_integrated


# =========================================================
# APP
# =========================================================
app = Flask(__name__)
CORS(app)  # Allow requests from the HTML frontend


# =========================================================
# ROUTES
# =========================================================
@app.route("/predict", methods=["POST"])
def predict():
    image_file = request.files.get("image")
    symptoms   = request.form.get("symptoms", "").strip()

    if not image_file:
        return jsonify({"error": "No image provided"}), 400

    if not symptoms:
        symptoms = "skin lesion"  # default fallback if user leaves blank

    # Save uploaded image to a temp file
    suffix = Path(image_file.filename).suffix if image_file.filename else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        image_file.save(tmp.name)
        tmp_path = Path(tmp.name)

    try:
        result = predict_integrated(tmp_path, symptoms)

        return jsonify({
            "disease":    result["final_prediction"],
            "confidence": result["final_topk"][0]["confidence"],
            "topk":       result["final_topk"],
            "image_topk": result["image_topk"],
            "nlp_topk":   result["nlp_topk"],
            "gradcam":    result["gradcam"]["output_path"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/gradcam-image", methods=["GET"])
def gradcam_image():
    """Serve the GradCAM heatmap image file to the frontend."""
    path = request.args.get("path")
    if not path:
        return jsonify({"error": "No path provided"}), 400

    gradcam_path = Path(path)
    if not gradcam_path.exists():
        return jsonify({"error": "GradCAM image not found"}), 404

    return send_file(str(gradcam_path), mimetype="image/png")


@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint."""
    return jsonify({"status": "ok", "message": "DermoAI API is running"})


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    print("=" * 50)
    print("DermoAI Flask API starting...")
    print(f"Project root: {PROJECT_ROOT}")
    print("API running at: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)