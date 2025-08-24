import pickle
import sys

def check_landmarks(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    total = len(data)
    valid = sum(1 for frame in data if frame is not None)
    print(f"Total frames: {total}")
    print(f"Frames with valid landmarks: {valid}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python backend/check_landmarks.py backend/landmarks/walking-video_landmarks.pkl")
        sys.exit(1)
    check_landmarks(sys.argv[1])
