**Crop Recommendation Model**
# 🌿 AgriFy - Crop Recommendation & Crop Rotation Chatbot

AgriFy is a smart, AI-powered crop recommendation system designed to assist farmers and agricultural professionals in choosing the best crop to cultivate based on soil and weather parameters. The project also includes a Gemini 1.5 Pro-powered chatbot to provide intelligent crop rotation suggestions.

---

## 🚀 Features
- 🔍 Predict the most suitable crop based on:
  - Nitrogen, Phosphorus, Potassium (NPK)
  - Temperature
  - Humidity
  - pH Level
  - Rainfall
- 💬 Gemini 1.5 Pro-powered Crop Rotation Chatbot
- 🖥️ Modern responsive UI built with Bootstrap 5
- 🔄 REST API integration using Flask
- 📊 Trained with Random Forest Classifier on labeled agricultural data

---

## 📁 Project Structure
```
AgriFy/
├── model/                       # Trained model storage
│   └── crop_recommender_model.joblib
├── templates/                  # HTML templates
│   ├── index.html
│   └── rotation.html
├── static/                     # CSS/JS (if any)
│   └── style.css (optional)
├── Crop_recommendation.csv     # Dataset
├── app.py                      # Flask application
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🧠 Tech Stack
- **Frontend:** HTML, CSS, Bootstrap 5, JavaScript
- **Backend:** Python, Flask
- **Machine Learning:** scikit-learn (Random Forest)
- **AI Chatbot:** Gemini 1.5 Pro via Google Generative AI
- **Model Persistence:** joblib

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/rajeev782/AgriFy-Crop_recommendation_model.git
cd AgriFy-Crop_recommendation_model
```

### 2️⃣ Create a Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Add Your Gemini API Key
In `app.py`, configure the Gemini model:
```python
genai.configure(api_key="YOUR_GEMINI_API_KEY")
```

### 5️⃣ Run the Flask App
```bash
python app.py
```

---

## 🤖 Model Training (Optional)
To retrain the model, modify and run the training block inside `app.py` or extract it into `train_model.py`.

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).

---

## ✨ Acknowledgements
- Dataset inspired by [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- Gemini 1.5 Pro by Google Generative AI

---

## 👨‍💻 Developed By
**Rajeev** — [@rajeev782](https://github.com/rajeev782)

Feel free to ⭐ this repo if you found it useful!
