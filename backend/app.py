import os
import json
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

from extract_gait import extract_gait_from_video
from calculate_gait_features import calc_all_gait_features
from ml_auth import train_rf_classifier, predict_user

app = Flask(__name__)
CORS(app)  # Allow all origins; restrict in production

PROFILE_DB = 'backend/user_profiles.json'

def load_profiles():
    if os.path.exists(PROFILE_DB):
        with open(PROFILE_DB, 'r') as f:
            return json.load(f)
    else:
        return {}

def save_profiles(profiles):
    with open(PROFILE_DB, 'w') as f:
        json.dump(profiles, f, indent=4)

@app.route('/register', methods=['POST'])
def register():
    user_id = request.form.get("user_id")
    video = request.files.get("video")

    if not user_id or not video:
        return jsonify({"error": "Missing user_id or video"}), 400

    # Directories
    video_dir = 'backend/uploaded_videos'
    landmarks_dir = 'backend/landmarks'
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(landmarks_dir, exist_ok=True)

    # Save uploaded video
    video_path = os.path.join(video_dir, f'{user_id}_video.webm')
    video.save(video_path)

    # Extract landmarks from video
    landmarks_path = os.path.join(landmarks_dir, f'{user_id}_landmarks.pkl')
    extract_gait_from_video(video_path, landmarks_path)

    # Load landmarks and calculate features
    with open(landmarks_path, 'rb') as f:
        landmarks = pickle.load(f)
    features = calc_all_gait_features(landmarks)

    # Load profiles and append features
    profiles = load_profiles()
    if user_id not in profiles:
        profiles[user_id] = []
    profiles[user_id].append(features)
    save_profiles(profiles)

    return jsonify({
        "message": "Registration successful",
        "user_id": user_id,
        "features": features
    }), 200

@app.route('/authenticate', methods=['POST'])
def authenticate():
    video = request.files.get("video")
    if not video:
        return jsonify({"error": "Missing video upload"}), 400

    # Save uploaded video
    auth_video_path = 'backend/uploaded_videos/auth_attempt.webm'
    os.makedirs(os.path.dirname(auth_video_path), exist_ok=True)
    video.save(auth_video_path)

    # Extract landmarks
    auth_landmarks_path = 'backend/landmarks/auth_attempt_landmarks.pkl'
    os.makedirs(os.path.dirname(auth_landmarks_path), exist_ok=True)
    extract_gait_from_video(auth_video_path, auth_landmarks_path)

    with open(auth_landmarks_path, 'rb') as f:
        landmarks = pickle.load(f)

    auth_features = calc_all_gait_features(landmarks)

    # Train classifier and predict user
    clf, scaler, feature_names = train_rf_classifier()
    pred_user, pred_conf = predict_user(auth_features, clf, scaler, feature_names)

    threshold = 0.6  # Tune as appropriate
    if pred_user  and pred_conf > threshold:
        return jsonify({
            "message": "Authentication successful",
            "matched_user": pred_user,
            "confidence": float(pred_conf),
            "features": auth_features
        }), 200
    else:
        return jsonify({
            "message": "No matching user found",
            "confidence": float(pred_conf),
            "features": auth_features
        }), 401

if __name__ == "__main__":
    app.run(debug=True)
