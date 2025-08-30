# 🩺 Breast Cancer Prediction App  

This is a **Flask-based web application** that predicts whether a breast tumor is **Malignant (Cancerous)** or **Benign (Non-Cancerous)** using patient tumor measurements.  
The model is trained on the **Breast Cancer Wisconsin (Diagnostic) Dataset**.  

---

## 📂 Project Structure  

CANCER_PREDICTION/
│
├── Dataset/
│ └── breast-cancer.csv # Dataset used for training the model
│
├── static/
│ ├── style.css # CSS for input form
│ └── result.css # CSS for result page
│
├── templates/
│ ├── index.html # Input form page
│ └── result.html # Prediction result page
│
├── app.py # Flask app (main entry point)
├── model_building.py # Model training script
├── model.pkl # Trained ML model
├── scaler.pkl # Scaler for preprocessing input data
├── requirements.txt # Dependencies for deployment
├── runtime.txt # Python version for Render deployment
└── README.md # Project documentation


---

## ⚙️ Installation  

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cancer_prediction.git
   cd cancer_prediction



2. Create a virtual environment:
python -m venv venv
source venv/bin/activate     # For Linux/Mac
venv\Scripts\activate        # For Windows


3. Install dependencies:
pip install -r requirements.txt

▶️ Usage

Run the Flask app:
python app.py


Open your browser and go to:
http://127.0.0.1:5000/


Enter tumor measurements in the form and click Predict.
Malignant → Cancerous tumor
Benign → Non-cancerous tumor



💡 Tech Stack

Python
Flask
Scikit-learn
NumPy / Pandas
HTML / CSS


✨ Future Improvements

Improve UI with Bootstrap or Tailwind CSS.
Add probability score with prediction.
Deploy as a full-stack app with React frontend.


👨‍💻 Author
Made with ❤️ by  Aaru 
