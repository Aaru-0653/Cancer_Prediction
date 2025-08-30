# app.py
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')   

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect input values from form
        try:
            features = [float(x) for x in request.form.values()]
            final_features = np.array(features).reshape(1, -1)

            # Scale input features
            final_features_scaled = scaler.transform(final_features)

            # Predict
            prediction = model.predict(final_features_scaled)
            result = "Malignant (Cancerous)" if prediction[0] == 1 else "Benign (Non-Cancerous)"

            return render_template('result.html', prediction=result)

        except Exception as e:
            return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
