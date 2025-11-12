import joblib
import pandas as pd
from feast import FeatureStore
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

PROJECT_ID = "winged-quanta-473915-m2"
DATASET = "iris_feast_ds"
TABLE = "iris_features"
SERVICE = "iris_service"

bq = bigquery.Client(project=PROJECT_ID)
store = FeatureStore(repo_path="./feast_featurestore/")

entity_sql = f"SELECT iris_id, event_timestamp FROM `{PROJECT_ID}.{DATASET}.{TABLE}`"
hist = store.get_historical_features(
    entity_df=entity_sql,
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width"
    ],
).to_df()


labels = bq.query(f"SELECT iris_id, event_timestamp, species FROM `{PROJECT_ID}.{DATASET}.{TABLE}`").result().to_dataframe()
df = hist.merge(labels, on=["iris_id","event_timestamp"], how="left")

X = df[["sepal_length","sepal_width","petal_length","petal_width"]].copy()
y = df["species"].map({"setosa":0, "versicolor":1, "virginica":2}).astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_tr, y_tr)
print(f"Accuracy: {accuracy_score(y_te, clf.predict(X_te)):.4f}")
joblib.dump(clf, "model_iris_bq.joblib")
print("Saved model_iris_bq.joblib")
