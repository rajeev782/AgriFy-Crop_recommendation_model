import pandas as pd
import joblib
import os
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = os.path.join(os.getcwd(), "Crop_recommendation.csv")  # Use absolute path
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
os.makedirs("model", exist_ok=True)  # Create model directory if it doesn't exist
model.fit(X_train_scaled, y_train)

# Save model and scaler
model_path = os.path.join("model", "crop_recommender_model.joblib")
joblib.dump({"model": model, "scaler": scaler}, model_path)

print("✅ Model trained and saved successfully!")

# Initialize Flask App
app = Flask(__name__)

# Load trained model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found: {model_path}")

model_data = joblib.load(model_path)
model = model_data["model"]
scaler = model_data["scaler"]

@app.route("/")
def home():
    return render_template("index.html")  # Ensure 'index.html' is inside 'templates/' folder

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("Received Input:", data)  # Debugging log

        input_data = pd.DataFrame([[data["N"], data["P"], data["K"], data["temperature"], 
                                    data["humidity"], data["ph"], data["rainfall"]]],
                                  columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        print("Predicted Crop:", prediction)  # Debugging log

        return jsonify({"crop": prediction})

    except Exception as e:
        print("Error:", str(e))  # Debugging log
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)  # Run Flask on port 5001
