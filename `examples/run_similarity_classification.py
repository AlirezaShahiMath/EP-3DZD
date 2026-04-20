import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

data = np.load("features/ep3dzd_features.npz", allow_pickle=True)
feat_dict = data["features_dict"].item()

X = np.array(list(feat_dict.values()))
y = np.random.randint(0, 2, len(X))

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

y_pred = model.predict_proba(X)[:, 1]

print("AUC:", roc_auc_score(y, y_pred))
