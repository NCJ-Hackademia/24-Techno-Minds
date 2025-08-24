import numpy as np
import pickle

# Landmark indices
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
HIP_LEFT = 23
KNEE_LEFT = 25
ANKLE_LEFT = 27
HIP_RIGHT = 24
KNEE_RIGHT = 26
ANKLE_RIGHT = 28
SHOULDER_LEFT = 11
SHOULDER_RIGHT = 12

def angle_between_points(a, b, c):
    """
    Calculate angle ABC in degrees given three 3D points a,b,c.
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calc_stride_and_stepfreq(landmarks, fps=30):
    left_ankle_positions = []
    right_ankle_positions = []
    frame_count = 0

    for frame in landmarks:
        if frame is None or len(frame) <= max(LEFT_ANKLE,RIGHT_ANKLE):
            continue
        left_ankle = frame[LEFT_ANKLE]
        right_ankle = frame[RIGHT_ANKLE]
        left_ankle_positions.append(left_ankle)
        right_ankle_positions.append(right_ankle)
        frame_count += 1

    if not left_ankle_positions or not right_ankle_positions:
        return 0, 0

    left = np.vstack(left_ankle_positions)
    right = np.vstack(right_ankle_positions)

    stride_length = np.mean(np.abs(left[:, 0] - right[:, 0]))
    crossing = np.diff(left[:, 0] > right[:, 0])
    cross_events = np.sum(crossing != 0)
    duration_seconds = frame_count / fps if frame_count > 0 else 1
    step_frequency = cross_events / duration_seconds

    return stride_length, step_frequency

def extract_angles(landmarks):
    knee_angles_left = []
    knee_angles_right = []
    hip_angles_left = []
    hip_angles_right = []

    for frame in landmarks:
        if frame is None or len(frame) <= max(HIP_LEFT, KNEE_LEFT, ANKLE_LEFT, HIP_RIGHT, KNEE_RIGHT, ANKLE_RIGHT, SHOULDER_LEFT, SHOULDER_RIGHT):
            continue
        knee_angles_left.append(angle_between_points(frame[HIP_LEFT], frame[KNEE_LEFT], frame[ANKLE_LEFT]))
        knee_angles_right.append(angle_between_points(frame[HIP_RIGHT], frame[KNEE_RIGHT], frame[ANKLE_RIGHT]))
        hip_angles_left.append(angle_between_points(frame[SHOULDER_LEFT], frame[HIP_LEFT], frame[KNEE_LEFT]))
        hip_angles_right.append(angle_between_points(frame[SHOULDER_RIGHT], frame[HIP_RIGHT], frame[KNEE_RIGHT]))

    def safe_stats(arr):
        return (np.mean(arr) if arr else 0, np.std(arr) if arr else 0)

    knee_left_mean, knee_left_std = safe_stats(knee_angles_left)
    knee_right_mean, knee_right_std = safe_stats(knee_angles_right)
    hip_left_mean, hip_left_std = safe_stats(hip_angles_left)
    hip_right_mean, hip_right_std = safe_stats(hip_angles_right)

    return {
        "knee_left_mean": knee_left_mean,
        "knee_right_mean": knee_right_mean,
        "hip_left_mean": hip_left_mean,
        "hip_right_mean": hip_right_mean,
        "knee_left_std": knee_left_std,
        "knee_right_std": knee_right_std,
        "hip_left_std": hip_left_std,
        "hip_right_std": hip_right_std,
    }

def calc_all_gait_features(landmarks, fps=30):
    stride_length, step_frequency = calc_stride_and_stepfreq(landmarks, fps)
    angle_features = extract_angles(landmarks)

    all_features = {
        "stride_length": stride_length,
        "step_frequency": step_frequency,
    }
    all_features.update(angle_features)
    return all_features

# Example usage (for testing standalone or in backend)
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python calculate_gait_features.py <landmarks.pkl>")
        sys.exit(1)

    landmarks_path = sys.argv[1]
    with open(landmarks_path, 'rb') as f:
        landmarks = pickle.load(f)

    features = calc_all_gait_features(landmarks)
    print("Extracted Gait Features:")
    for k, v in features.items():
        print(f"{k}: {v:.4f}")
