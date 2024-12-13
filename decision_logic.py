def calculate_confidence(pose_landmarks, emotion):
    confidence_score = 0

    if emotion in ['happy', 'neutral']:
        confidence_score += 50
    if pose_landmarks:  # Check if pose landmarks are detected
        confidence_score += 50

    return confidence_score

def make_decision(confidence_score):
    if confidence_score >= 75:
        return "Approved"
    else:
        return "Needs Manual Review"


def generate_summary(metrics):
    total_frames = metrics['total_frames']
    total_duration = metrics.get('total_duration', 0)

    # Avoid division by zero
    nervousness_percentage = (metrics['nervousness_score'] / total_frames) * 100 if total_frames else 0
    posture_issues_percentage = (metrics['posture_issues'] / total_frames) * 100 if total_frames else 0

    # Determine dominant emotion
    dominant_emotion = max(metrics['emotion_count'], key=metrics['emotion_count'].get)

    # Debugging
    print("\nFinal Metrics for Summary:")
    print(metrics)

    # Generate summary
    summary = {
        'total_duration': round(total_duration, 2),  # Round to 2 decimal places
        'dominant_emotion': dominant_emotion,
        'nervousness_percentage': round(nervousness_percentage, 2),
        'posture_issues_percentage': round(posture_issues_percentage, 2),
        'final_decision': 'Confident' if nervousness_percentage < 30 else 'Nervous',
        'dress_code': "Approved" if metrics['dress'] else "Not Approved"
    }
    return summary
