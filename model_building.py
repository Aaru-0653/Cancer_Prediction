# model_building.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load dataset
df = pd.read_csv("Dataset/breast-cancer.csv")  

# 2. Drop useless column (id) and encode target (diagnosis)
df = df.drop(columns=["id"])
df["diagnosis"] = LabelEncoder().fit_transform(df["diagnosis"])  
# M -> 1 (Malignant), B -> 0 (Benign)

# 3. Split features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# 4. Standardize features (important for ML performance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6. Build and train model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 8. Save model and scaler for Flask app
with open("model.pkl", "wb") as f:
    joblib.dump(model, f)

with open("scaler.pkl", "wb") as f:
    joblib.dump(scaler, f)
