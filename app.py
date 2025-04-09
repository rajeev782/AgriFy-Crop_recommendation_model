import pandas as pd
import joblib
import os
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
from flask_cors import CORS

# Load dataset
file_path = os.path.join(os.getcwd(), "Crop_recommendation.csv")
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found: {file_path}")

df = pd.read_csv(file_path)

# Split features and target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=20)
os.makedirs("model", exist_ok=True)
model.fit(X_train_scaled, y_train)

# Save model and scaler
model_path = os.path.join("model", "crop_recommender_model.joblib")
joblib.dump({"model": model, "scaler": scaler}, model_path)

print("✅ Model trained and saved successfully!")

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Load trained model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found: {model_path}")

model_data = joblib.load(model_path)
model = model_data["model"]
scaler = model_data["scaler"]

# Configure Gemini API
genai.configure(api_key="AIzaSyA2jO0bCK5dSuqkKO2ZoXOyXTa1DQTKe9M")
gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")
chat = gemini_model.start_chat(history=[])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("Received Input:", data)

        input_data = pd.DataFrame([[data["N"], data["P"], data["K"], data["temperature"],
                                    data["humidity"], data["ph"], data["rainfall"]]],
                                  columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        print("Predicted Crop:", prediction)

        return jsonify({"crop": prediction})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)})

@app.route("/rotation")
def rotation_page():
    return render_template("rotation.html")

@app.route("/ask_rotation", methods=["POST"])
def ask_rotation():
    try:
        data = request.get_json()
        user_query = data.get("question", "")

        print("📥 User asked:", user_query)

        response = chat.send_message(user_query)
        raw_reply = response.text

        print("📤 Gemini replied:", raw_reply)

        html_reply = raw_reply.replace("**", "").replace("*", "").replace("\n", "<br>")

        return jsonify({"answer": html_reply})
    except Exception as e:
        print("❌ Error in Gemini:", str(e))
        return jsonify({"answer": "Sorry, something went wrong: " + str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
