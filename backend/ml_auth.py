import numpy as np
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

PROFILE_DB = 'backend/user_profiles.json'

def load_training_data():
    if not os.path.exists(PROFILE_DB):
        return [], [], []
    with open(PROFILE_DB, 'r') as f:
        profiles = json.load(f)
    X = []
    y = []
    feature_names = []
    for user_id, samples in profiles.items():
        for feature_dict in samples:
            if not feature_names:
                feature_names = sorted(feature_dict.keys())
            X.append([feature_dict[k] for k in feature_names])
            y.append(user_id)
    return np.array(X), np.array(y), feature_names

def train_rf_classifier(n_estimators=100):
    X, y, feature_names = load_training_data()
    if len(X) == 0:
        return None, None, feature_names

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_scaled, y)

    if len(X) >= 5:
        scores = cross_val_score(clf, X_scaled, y, cv=5)
        print(f'Random Forest 5-fold CV Accuracy: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})')
    else:
        print('Not enough samples for cross-validation.')

    return clf, scaler, feature_names

def predict_user(features, clf, scaler, feature_names):
    if clf is None or scaler is None or not feature_names:
        return None, 0.0
    X_test = np.array([[features[k] for k in feature_names]])
    X_test_scaled = scaler.transform(X_test)
    pred = clf.predict(X_test_scaled)[0]
    proba = clf.predict_proba(X_test_scaled).max() if hasattr(clf, "predict_proba") else 1.0
    return pred, proba
