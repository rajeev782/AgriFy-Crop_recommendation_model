import pandas as pd
import joblib
import os
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Model Training (Only if model file doesn't exist) ---
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "crop_recommender.joblib")
DATASET_PATH = os.path.join(os.getcwd(), "Crop_recommendation.csv")

if not os.path.exists(MODEL_PATH):
    print("Model not found. Starting training process...")

    # Ensure dataset exists
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    # Load and prepare data
    df = pd.read_csv(DATASET_PATH)
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=20)
    model.fit(X_train_scaled, y_train)

    # Save model and scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)

    print("✅ Model trained and saved successfully!")
else:
    print("✅ Model already exists. Loading from disk.")


# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# Load the trained model and scaler
model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
scaler = model_data["scaler"]

# Configure Gemini API using the securely loaded key
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")
    chat = gemini_model.start_chat(history=[])
    print("✅ Gemini API configured successfully.")
except Exception as e:
    print(f"❌ Error configuring Gemini API: {e}")
    chat = None

# --- API Endpoints ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        input_data = pd.DataFrame([[data["N"], data["P"], data["K"], data["temperature"],
                                    data["humidity"], data["ph"], data["rainfall"]]],
                                  columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        return jsonify({"crop": prediction})

    except Exception as e:
        print(f"❌ Prediction Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/rotation")
def rotation_page():
    return render_template("rotation.html")

@app.route("/ask_rotation", methods=["POST"])
def ask_rotation():
    if not chat:
        return jsonify({"answer": "Sorry, the chatbot is not available due to a configuration error."}), 503

    try:
        data = request.get_json()
        user_query = data.get("question", "")
        
        print(f"📥 User asked: {user_query}")
        
        response = chat.send_message(user_query)
        raw_reply = response.text
        
        print(f"📤 Gemini replied: {raw_reply}")
        
        # Basic formatting for HTML display
        html_reply = raw_reply.replace("**", "<b>").replace("*", "<i>").replace("\n", "<br>")

        return jsonify({"answer": html_reply})
        
    except Exception as e:
        print(f"❌ Gemini Error: {str(e)}")
        return jsonify({"answer": "Sorry, something went wrong with the chatbot: " + str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)