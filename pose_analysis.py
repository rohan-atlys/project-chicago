import mediapipe as mp
import cv2
import numpy as np
import time

class InterviewPoseAnalyzer:
    def __init__(self, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize Interview Pose Analyzer with more relaxed detection
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=0  # Lower complexity for more forgiving detection
        )

    def calculate_angle(self, a, b, c):
        """
        Calculate angle between three points
        """
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        return angle if angle <= 180 else 360 - angle

    def analyze_interview_posture(self, frame):
        """
        Analyze upper body posture for an interview setting with more relaxed criteria
        """
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame and detect pose
        results = self.pose.process(frame_rgb)
        
        # Initialize analysis results dictionary
        posture_analysis = {
            'detected': False,
            'professional_score': 0,
            'posture_issues': [],
            'body_angles': {}
        }
        
        # If landmarks are detected
        if results.pose_landmarks:
            posture_analysis['detected'] = True
            landmarks = results.pose_landmarks.landmark
            
            try:
                # Shoulder alignment and symmetry (more lenient)
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]
                shoulder_width_score = self._check_shoulder_alignment(left_shoulder, right_shoulder)
                
                # Head and neck posture (wider acceptable range)
                head_posture_score = self._analyze_head_posture(landmarks)
                
                # Upper body orientation (more flexible)
                upper_body_orientation = self._check_upper_body_orientation(landmarks)
                
                # Arm positioning (less strict)
                arm_positioning_score = self._analyze_arm_position(landmarks)
                
                # Compile professional posture score with weighted averaging
                posture_analysis['professional_score'] = (
                    (shoulder_width_score * 0.2) + 
                    (head_posture_score * 0.3) + 
                    (upper_body_orientation * 0.3) + 
                    (arm_positioning_score * 0.2)
                )
                
                # Generate specific posture feedback
                self._generate_posture_feedback(posture_analysis, shoulder_width_score, 
                                                head_posture_score, upper_body_orientation, 
                                                arm_positioning_score)
                
                # Draw landmarks for visualization
                self.mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS
                )
                
            except Exception as e:
                print(f"Error in interview posture analysis: {e}")
        
        return frame, posture_analysis

    def _check_shoulder_alignment(self, left_shoulder, right_shoulder):
        shoulder_angle = self.calculate_angle(
            type('Point', (), {'x': left_shoulder.x - 0.1, 'y': left_shoulder.y}),
            left_shoulder,
            right_shoulder
        )
        
        # Relax shoulder alignment range
        if 160 <= shoulder_angle <= 200:  # Increased the range
            return 1.0
        elif 150 <= shoulder_angle <= 210:  # Increased the range
            return 0.8
        elif 140 <= shoulder_angle <= 220:  # Increased the range
            return 0.6
        else:
            return 0.4

    def _analyze_head_posture(self, landmarks):
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        # Calculate head tilt and forward lean
        head_tilt_angle = self.calculate_angle(left_shoulder, nose, right_shoulder)
        
        # Relax head tilt range
        if 50 <= head_tilt_angle <= 130:  # Expanded range
            return 1.0
        elif 40 <= head_tilt_angle <= 140:  # Expanded range
            return 0.8
        elif 30 <= head_tilt_angle <= 150:  # Expanded range
            return 0.6
        else:
            return 0.4

    def _check_upper_body_orientation(self, landmarks):
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Calculate spine angle
        spine_angle = self.calculate_angle(left_shoulder, left_hip, right_hip)
        
        # Relax spine angle range
        if 150 <= spine_angle <= 210:  # Expanded range
            return 1.0
        elif 140 <= spine_angle <= 220:  # Expanded range
            return 0.8
        elif 130 <= spine_angle <= 230:  # Expanded range
            return 0.6
        else:
            return 0.4

    def _analyze_arm_position(self, landmarks):
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_elbow = landmarks[13]
        right_elbow = landmarks[14]
        
        # Relax arm positioning ranges
        left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, 
            type('Point', (), {'x': left_elbow.x + 0.1, 'y': left_elbow.y}))
        right_arm_angle = self.calculate_angle(right_shoulder, right_elbow, 
            type('Point', (), {'x': right_elbow.x + 0.1, 'y': right_elbow.y}))
        
        if 40 <= left_arm_angle <= 140 and 40 <= right_arm_angle <= 140:  # Relaxed angle ranges
            return 1.0
        elif 30 <= left_arm_angle <= 150 and 30 <= right_arm_angle <= 150:  # Relaxed angle ranges
            return 0.8
        elif 20 <= left_arm_angle <= 160 and 20 <= right_arm_angle <= 160:  # Relaxed angle ranges
            return 0.6
        else:
            return 0.4

    def _generate_posture_feedback(self, posture_analysis, shoulder_score, 
                                   head_score, body_orientation, arm_score):
        """
        Generate more constructive and less critical posture feedback
        """
        posture_analysis['posture_issues'] = []
        
        if shoulder_score < 0.7:
            posture_analysis['posture_issues'].append("Consider slight shoulder adjustment")
        
        if head_score < 0.7:
            posture_analysis['posture_issues'].append("Try to maintain a comfortable head position")
        
        if body_orientation < 0.7:
            posture_analysis['posture_issues'].append("Aim for a relaxed, upright posture")
        
        if arm_score < 0.7:
            posture_analysis['posture_issues'].append("Keep arms naturally positioned")

    def visualize_interview_posture(self, frame, posture_analysis):
        """
        Visualize interview posture analysis results with more encouraging tone
        """
        if not posture_analysis['detected']:
            cv2.putText(frame, "Pose not clearly detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            return frame
        
        # Professional score color coding (more gradual)
        score_color = (0, 255, 0)  # Green
        if posture_analysis['professional_score'] < 0.4:
            score_color = (0, 0, 255)  # Red
        elif posture_analysis['professional_score'] < 0.7:
            score_color = (0, 165, 255)  # Orange
        
        # Display professional score with encouraging message
        score_text = f"Posture Score: {posture_analysis['professional_score']*100:.1f}%"
        cv2.putText(frame, score_text, 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    score_color, 
                    2)
        
        # Display posture suggestions more gently
        if posture_analysis['posture_issues']:
            suggestions_text = ", ".join(posture_analysis['posture_issues'])
            cv2.putText(frame, f"Suggestions: {suggestions_text}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 255), 1)
        
        return frame

# Example usage remains the same as in the previous version
def main():
    cap = cv2.VideoCapture(0)
    interview_analyzer = InterviewPoseAnalyzer()
    
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # analyse frame every 5th second
        current_time = time.time()
        if int(current_time - start_time) % 5 == 0:
            annotated_frame, posture_analysis = interview_analyzer.analyze_interview_posture(frame)
            result_frame = interview_analyzer.visualize_interview_posture(annotated_frame, posture_analysis)
        else:
            result_frame = frame

        cv2.imshow('Interview Posture Analysis', result_frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()