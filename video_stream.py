import cv2
from pose_analysis import  InterviewPoseAnalyzer
from emotion_analysis import analyze_emotion
from decision_logic import generate_summary
from dress_analyzer import analyze_dress
import os
import time

import threading

cv_loop_event = threading.Event()

def generate_frames(metrics):
    os.makedirs("captured_frames", exist_ok=True)

    camera = cv2.VideoCapture(0)
    start_time = time.time()  # Record the start time

    pose_analyzer = InterviewPoseAnalyzer()

    image_saved = False
    try:
        cv_loop_event.set()
        while cv_loop_event.is_set():
            success, frame = camera.read()
            if not success:
                break

            metrics['total_frames'] += 1

            # Analyze pose
            frame, pose_landmarks = pose_analyzer.analyze_interview_posture(frame)

            # Analyze emotion every 5th second
            current_time = time.time()
            if int(current_time - start_time) % 5 == 0:
                # save two frames for dress analysis later on
                if metrics['total_frames'] > 2 and not image_saved:
                    filename = os.path.join("captured_frames", f'frame_{int(current_time)}.jpg')
                    cv2.imwrite(filename, frame)
                    image_saved = True
                
                emotion = analyze_emotion(frame)
                if emotion in metrics['emotion_count']:
                    metrics['emotion_count'][emotion] += 1

                # Check for nervousness or posture issues
                if emotion == 'nervous' or not pose_landmarks:
                    metrics['nervousness_score'] += 1
                if not pose_landmarks:
                    metrics['posture_issues'] += 1

            # Encode the frame for streaming
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Ending session.")
    finally:
        # Calculate total duration
        end_time = time.time()
        metrics['total_duration'] = end_time - start_time  # Calculate total duration in seconds
        camera.release()
        cv2.destroyAllWindows()
