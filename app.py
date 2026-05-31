"""
FallGuard — Flask Backend
=========================
Before running this file, save your trained model from the Colab notebook:

    model_lstm.save("model.h5")          # or model_lstm.save("model_saved")

Then place model.h5 (or model_saved/) in the same folder as this file.
Run with:  python app.py
Open:      http://127.0.0.1:5000
"""

import os
import uuid
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf

# ──────────────────────────────────────────────
# CONFIG — change MODEL_PATH if needed
# ──────────────────────────────────────────────
MODEL_PATH     = "model.h5"       # or "model_saved" for SavedModel format
SEQUENCE_LEN   = 5
IMG_SIZE       = 224
THRESHOLD      = 0.5             # best threshold from your notebook experiments
UPLOAD_FOLDER  = "uploads"
# ──────────────────────────────────────────────

app = Flask(__name__, static_folder=".", static_url_path="")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once at startup (not on every request)
print("Loading model from", MODEL_PATH, "...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded. Input shape:", model.input_shape)


def extract_one_window(cap, start_frame, end_frame, sequence_length, img_size):
    """Extract one sequence of frames between start_frame and end_frame."""
    indices = np.linspace(start_frame, end_frame - 1, sequence_length, dtype=int)
    sequence = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size))
        frame = frame.astype("float32") / 255.0
        sequence.append(frame)
    return sequence


def extract_all_windows(video_path, sequence_length=SEQUENCE_LEN, img_size=IMG_SIZE):
    """
    Run 3 sampling windows across the video and return all of them.

    Why: During training, fall videos were sampled from the LAST 20% of frames
    (where the fall actually happens). At inference we don't know where the fall
    is, so we check early / middle / late windows and take the max prediction.

    Windows:
      - Early:  first  50% of clip
      - Middle: middle 50% of clip
      - Late:   last   50% of clip  ← most important, mirrors training
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < sequence_length:
        cap.release()
        raise ValueError(
            f"Video has only {total_frames} frames — need at least {sequence_length}."
        )

    half = total_frames // 2

    windows = {
    	"last20": (int(total_frames * 0.8), total_frames),  # exactly matches training!
    	"last50": (total_frames // 2,        total_frames),
    	"full":   (0,                         total_frames),
      }

    all_sequences = {}
    for name, (start, end) in windows.items():
        seq = extract_one_window(cap, start, end, sequence_length, img_size)
        if len(seq) >= sequence_length:
            all_sequences[name] = np.expand_dims(np.array(seq[:sequence_length]), axis=0)

    cap.release()

    if not all_sequences:
        raise ValueError("Could not read enough frames from the video.")

    return all_sequences


def build_response(probability, threshold=THRESHOLD):
    """
    Convert a raw sigmoid probability into a structured prediction dict.
    """
    is_fall     = float(probability) >= threshold
    confidence  = float(probability) if is_fall else 1.0 - float(probability)
    label       = "Fall Detected" if is_fall else "No Fall Detected"

    if is_fall:
        actions = [
            "Check on the person immediately — confirm responsiveness.",
            "Call emergency services (000) if the person is unresponsive.",
            "Do not move the person if a spinal or head injury is suspected; keep them still until help arrives.",
        ]
    else:
        actions = [
            "No immediate action required.",
            "Continue monitoring the area.",
        ]

    return {
        "label":       label,
        "is_fall":     is_fall,
        "probability": round(float(probability), 4),
        "confidence":  round(confidence * 100, 1),   # as percentage
        "threshold":   threshold,
        "actions":     actions,
    }


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend HTML file."""
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts: multipart/form-data with a 'video' field
    Returns: JSON prediction result
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file received."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # Save temporarily
    ext       = os.path.splitext(file.filename)[1].lower() or ".mp4"
    tmp_name  = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}{ext}")
    file.save(tmp_name)

    try:
        windows = extract_all_windows(tmp_name)

        # Run prediction on each window, take the highest score.
        # A single window catching a fall is enough to flag it.
        scores = {}
        for name, seq in windows.items():
            prob = float(model.predict(seq, verbose=0)[0][0])
            scores[name] = round(prob, 4)
            print(f"  [{name}] probability: {prob:.4f}")

        best_prob   = max(scores.values())
        best_window = max(scores, key=scores.get)
        print(f"  Best window: {best_window} ({best_prob:.4f})")

        result = build_response(best_prob)
        result["window_scores"] = scores
        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    finally:
        # Clean up the temp file
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


if __name__ == "__main__":
    print("\nFallGuard is running at http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
