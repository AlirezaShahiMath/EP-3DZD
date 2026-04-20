#!/usr/bin/env python
"""
Run binding affinity regression (EP-3DZD / EIM / Combined)
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

def load_features(path):
    data = np.load(path, allow_pickle=True)
    return data['features_dict'].item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--labels', type=str, required=True)
    args = parser.parse_args()

    feat_dict = load_features(args.features)
    labels_df = pd.read_csv(args.labels)

    X, y = [], []

    for _, row in labels_df.iterrows():
        pdbid = row['PDBID']
        if pdbid in feat_dict:
            X.append(feat_dict[pdbid])
            y.append(row['pK'])

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
