import pandas as pd

# Load player+cluster dataset
mp = pd.read_csv("clean_dataset/match_players_with_clusters.csv")

# Aggregated team features
team_features = (
    mp.groupby(["match_id", "team"])
    .agg({
        # Role composition counts
        "cluster": lambda x: x.value_counts().to_dict(),

        # Performance metrics aggregated by average
        "adr": "mean",
        "survival_rate": "mean",
        "kd_ratio": "mean",
        "first_kills": "sum",
        "first_deaths": "sum",
        "multi_kill_rounds": "sum",
        "flash_assists": "sum",
        "utility_damage": "mean",
        "performance_score": "mean",
        "won_match": "max"
    })
    .reset_index()
)

team_features.head()

# Expand cluster dict into separate columns per cluster type
cluster_df = team_features["cluster"].apply(lambda d: pd.Series(d)).fillna(0)
cluster_df.columns = [f"cluster_{int(c)}" for c in cluster_df.columns]

team_data = pd.concat([team_features.drop(columns=["cluster"]), cluster_df], axis=1)

team_data.head()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

features = [
    "adr", "survival_rate", "kd_ratio",
    "first_kills", "first_deaths",
    "multi_kill_rounds", "flash_assists",
    "utility_damage", "performance_score",
    "cluster_0", "cluster_1", "cluster_2",
    "cluster_3", "cluster_4", "cluster_5"
]

X = team_data[features]
y = team_data["won_match"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

import numpy as np

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

for idx in indices:
    print(f"{features[idx]}: {importances[idx]:.4f}")

