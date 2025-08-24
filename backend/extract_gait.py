import cv2
import mediapipe as mp
import pickle

def extract_gait_from_video(video_path, output_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)
    all_landmarks = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            all_landmarks.append(landmarks)
        else:
            all_landmarks.append(None)
    cap.release()
    # Save all landmark data to a file
    with open(output_path, "wb") as f:
        pickle.dump(all_landmarks, f)
    print(f"Landmarks saved to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("How to run:\npython backend/extract_gait.py backend/uploaded_videos/walking-video.webm backend/landmarks/walking-video_landmarks.pkl")
        sys.exit(1)
    video_file = sys.argv[1]
    output_file = sys.argv[2]
    extract_gait_from_video(video_file, output_file)
