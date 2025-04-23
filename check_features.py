import joblib

try:
    feature_columns = joblib.load("feature_columns.pkl")
    print("✅ feature_columns loaded:", feature_columns)
    print("📦 Type:", type(feature_columns))
except Exception as e:
    print("❌ Failed to load feature_columns.pkl:", e)