from flask import Flask, Response, render_template, jsonify  # Ensure render_template is imported
from video_stream import generate_frames, cv_loop_event
from decision_logic import generate_summary
from dress_analyzer import analyze_dress
import os
import base64


app = Flask(__name__)



# Global metrics
metrics = {
    'emotion_count': {'happy': 0, 'neutral': 0, 'nervous': 0},
    'posture_issues': 0,
    'total_frames': 0,
    'nervousness_score': 0,
    "dress": None
}

@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML template

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(metrics), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/end_interview', methods=['POST'])
def end_interview():
    global metrics

    cv_loop_event.clear()
    base_64_images = []
    if os.path.exists("captured_frames"):
        images = [os.path.join("captured_frames", img) for img in os.listdir("captured_frames")]
        for img in images:
            with open(img, "rb") as image_file:
                base_64_images.append(base64.b64encode(image_file.read()).decode('utf-8'))

    if not base_64_images:
        return jsonify({"error": "No images captured for dress analysis"})

    # delete captured frames
    for img in os.listdir("captured_frames"):
        os.remove(os.path.join("captured_frames", img))

    analyse_dress = analyze_dress(base_64_images)
    
    metrics['dress'] = analyse_dress
    summary = generate_summary(metrics)

    print("\nMetrics at the end of the interview:")
    print(metrics)
    # Reset metrics for the next session
    metrics = {
        'emotion_count': {'happy': 0, 'neutral': 0, 'nervous': 0},
        'posture_issues': 0,
        'total_frames': 0,
        'nervousness_score': 0,
        'total_duration': 0,
        'dress': None
    }
 
    return jsonify(summary)


if __name__ == "__main__":
    app.run(debug=True)
