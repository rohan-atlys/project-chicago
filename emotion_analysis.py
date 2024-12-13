from deepface import DeepFace
import cv2

def analyze_emotion(frame):
    try:
        # Convert frame to RGB (DeepFace requires RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Analyze emotions
        result = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)

        # Handle if result is a list (multiple detections)
        if isinstance(result, list):
            # Take the first result if multiple faces are detected
            result = result[0]

        # Get the dominant emotion
        dominant_emotion = result.get('dominant_emotion', 'Unknown')
        print(f"Detected Emotions: {result['emotion']}")  # Log all emotions with confidence
        return dominant_emotion

    except Exception as e:
        print(f"Emotion detection error: {e}")
        return "Unknown"
