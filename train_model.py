import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_excel("surgery recovery dataset.xlsx")

# Remove patient name
df.drop(columns=["Patient_Name"], inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in ["Gender", "Type_of_Surgery", "Exercise_Level"]:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop("Recovery_Days", axis=1)
y = df["Recovery_Days"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=150, learning_rate=0.1)
}

best_model = None
best_score = -1

# Train and select best model
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)

    print(f"{name} R2 Score: {score:.3f}")

    if score > best_score:
        best_score = score
        best_model = model

# Save model
joblib.dump(best_model, "model.pkl")

print("✅ model.pkl generated successfully!")
