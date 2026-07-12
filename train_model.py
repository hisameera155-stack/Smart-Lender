import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("loan_data.csv")

# Remove unwanted index column
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Remove Loan_ID if present
if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)
X=df.drop(TARGET_COLUMN, axis=1)
print("Features:", X.columns.tolist())
print("Number of features:", len(X.columns))

# -----------------------------
# Handle Missing Values
# -----------------------------
# Fill numeric columns
numeric_cols = df.select_dtypes(include=["number"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical/string columns
categorical_cols = df.select_dtypes(exclude=["number"]).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
# -----------------------------
# Encode Categorical Columns
# -----------------------------
encoders = {}

for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# -----------------------------
# Define Features & Target
# -----------------------------
TARGET_COLUMN = "Loan_Status"   # Change if your target column has a different name

X = df.drop(TARGET_COLUMN, axis=1)
print(X.columns.tolist())
y = df[TARGET_COLUMN]
print(df["Loan_Status"].value_counts())
# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,

)

# -----------------------------
# Train XGBoost Model
# -----------------------------
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# Save Model
# -----------------------------
with open("loan_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save Label Encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("Model saved as loan_model.pkl")
print("Encoders saved as label_encoders.pkl")