from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import logging
import os
import numpy as np
from pathlib import Path

app = Flask(__name__)

# CORS - Restrict in production to your frontend domains
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"])  # Change "*" for production

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ====================== MODEL LOADING ======================
MODEL_FILENAME = "cfoptimized_customer_payment_model.pkl"
MODEL_PATH = Path(MODEL_FILENAME)

loaded_model = None
RESIDUAL_STD = None   # Will hold the standard deviation of residuals for confidence

try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please place it in the same directory.")

    loaded_model = joblib.load(MODEL_PATH)
    logger.info(f"✅ Model loaded successfully: {MODEL_PATH}")

    # =============== SET YOUR RESIDUAL STD HERE ===============
    # Best practice: Compute this from your training data like this:
    # residuals = np.abs(y_train - loaded_model.predict(X_train))
    # RESIDUAL_STD = np.std(residuals)          # or np.sqrt(mean_squared_error(y_train, predictions))
    #
    # For now, set a realistic value based on your payment scale (e.g. $100–$400)
    RESIDUAL_STD = 185.0   # ←←← UPDATE THIS WITH YOUR ACTUAL VALUE

    logger.info(f"✅ Using residual std = ${RESIDUAL_STD:.2f} for 90% prediction intervals")

except Exception as e:
    logger.error(f"❌ Failed to load model or residuals: {e}")
    raise

# ====================== ROUTES ======================
@app.route('/')
def index():
    return render_template('prediction.html')

@app.route('/health')
def health():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": loaded_model is not None,
        "residual_std": round(RESIDUAL_STD, 2) if RESIDUAL_STD else None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid or missing JSON in request body"}), 400

        # Validation
        required = ["previous_month_payment", "two_months_ago_payment"]
        missing = [field for field in required if field not in data]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400

        # Convert and validate numbers
        try:
            prev_payment = float(data["previous_month_payment"])
            two_months_payment = float(data["two_months_ago_payment"])

            if prev_payment < 0 or two_months_payment < 0:
                return jsonify({"error": "Payment amounts cannot be negative"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Both fields must be valid numbers"}), 400

        logger.info(f"Prediction request → Prev: ${prev_payment:.2f}, 2mo ago: ${two_months_payment:.2f}")

        # Prepare features (must match training column names exactly)
        features = pd.DataFrame({
            "Previous Month Payment": [prev_payment],
            "Two Months Ago Payment": [two_months_payment]
        })

        # Make point prediction
        raw_prediction = loaded_model.predict(features)[0]
        predicted_payment = round(float(raw_prediction), 2)

        # Calculate 90% prediction interval using residual std
        if RESIDUAL_STD is not None:
            margin = 1.645 * RESIDUAL_STD          # 1.645 ≈ z-score for 90% confidence (two-tailed)
            lower_bound = round(predicted_payment - margin, 2)
            upper_bound = round(predicted_payment + margin, 2)
        else:
            lower_bound = upper_bound = None

        response = {
            "predicted_payment": predicted_payment,
            "confidence_level": "90%",
            "status": "success"
        }

        if lower_bound is not None:
            response.update({
                "confidence_lower": lower_bound,
                "confidence_upper": upper_bound
            })

        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 'yes')
    port = int(os.environ.get('PORT', 5000))

    if debug_mode:
        logger.warning("🚨 Running in DEBUG mode — not recommended for production!")

    logger.info(f"Starting Flask server on http://0.0.0.0:{port}")
    app.run(
        debug=debug_mode,
        host="0.0.0.0",
        port=port,
        threaded=True
    )