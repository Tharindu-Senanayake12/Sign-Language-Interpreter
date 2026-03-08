import sys
import time
import math
from collections import deque
import cv2 as cv
import numpy as np
import mediapipe as mp

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QFrame, 
                               QGraphicsDropShadowEffect, QSizePolicy, QComboBox)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap, QColor, QFont, QCursor
from PySide6.QtTextToSpeech import QTextToSpeech

class RehabPoseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # --- EXERCISE STATE & METRICS ---
        self.is_exercising = False
        self.rep_count = 0
        self.current_stage = "down" 
        self.feedback_text = "Select an exercise and press Start."
        self.feedback_color = (255, 255, 255)
        self.last_speech_time = 0
        
        # Buffer to smooth out angle jitter (Rolling Average)
        self.angle_buffer = deque(maxlen=5) 
        
        # --- TEXT TO SPEECH ---
        self.tts = QTextToSpeech(self)
        
        # --- MEDIAPIPE POSE (UPGRADED ACCURACY) ---
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.85, # Increased strictness
            min_tracking_confidence=0.85,  # Increased strictness
            model_complexity=2             # 2 is the heaviest, most accurate 3D model
        )

        self.init_ui()
        
        # --- CAMERA SETUP ---
        self.cap = cv.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.main_loop)
        self.timer.start(30) 

    def init_ui(self):
        """Constructs the modern PySide6 UI."""
        self.setWindowTitle("Rehab AR Assistant | Precision HPE")
        self.setMinimumSize(1300, 850)
        self.setStyleSheet("background-color: #F2F2F7;")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setContentsMargins(40, 40, 40, 40)
        self.main_layout.setSpacing(40)

        # -- LEFT: AR VIDEO FEED --
        left_vbox = QVBoxLayout()
        title = QLabel("Rehab AR Assistant")
        title.setStyleSheet("font-size: 38px; font-weight: 800; color: #1C1C1E; letter-spacing: -1.5px;")
        left_vbox.addWidget(title)

        self.video_frame = QFrame()
        self.video_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_frame.setStyleSheet("background-color: #000000; border-radius: 30px;")
        
        v_layout = QVBoxLayout(self.video_frame)
        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignCenter)
        v_layout.addWidget(self.video_display)
        
        shadow = QGraphicsDropShadowEffect(blurRadius=60, xOffset=0, yOffset=25, color=QColor(0,0,0,40))
        self.video_frame.setGraphicsEffect(shadow)
        left_vbox.addWidget(self.video_frame, stretch=1)
        self.main_layout.addLayout(left_vbox, stretch=3)

        # -- RIGHT: CONTROLS & METRICS --
        right_vbox = QVBoxLayout()
        right_vbox.setSpacing(25)

        selection_label = QLabel("Target Exercise")
        selection_label.setStyleSheet("font-size: 18px; font-weight: 700; color: #8E8E93;")
        right_vbox.addWidget(selection_label)

        self.primary_combo = QComboBox()
        self.primary_combo.addItems(["Select Region...", "Full Body", "Hands (Upper Body)", "Legs (Lower Body)"])
        self.primary_combo.setStyleSheet(self.combo_style())
        self.primary_combo.currentTextChanged.connect(self.update_sub_combo)
        right_vbox.addWidget(self.primary_combo)

        self.secondary_combo = QComboBox()
        self.secondary_combo.addItem("Waiting for region selection...")
        self.secondary_combo.setStyleSheet(self.combo_style())
        right_vbox.addWidget(self.secondary_combo)

        self.toggle_btn = QPushButton("START EXERCISE")
        self.toggle_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.toggle_btn.setStyleSheet("""
            QPushButton { background-color: #34C759; color: white; font-size: 16px; 
                          font-weight: 800; border-radius: 20px; padding: 18px; }
            QPushButton:hover { background-color: #2EAF4E; }
        """)
        self.toggle_btn.clicked.connect(self.toggle_exercise)
        right_vbox.addWidget(self.toggle_btn)

        self.metrics_card = QFrame()
        self.metrics_card.setStyleSheet("background-color: white; border-radius: 25px;")
        metrics_shadow = QGraphicsDropShadowEffect(blurRadius=20, xOffset=0, yOffset=10, color=QColor(0,0,0,15))
        self.metrics_card.setGraphicsEffect(metrics_shadow)
        
        m_layout = QVBoxLayout(self.metrics_card)
        m_layout.setContentsMargins(30, 30, 30, 30)
        
        rep_title = QLabel("REPETITIONS")
        rep_title.setStyleSheet("color: #8E8E93; font-size: 14px; font-weight: 800;")
        self.rep_label = QLabel("0")
        self.rep_label.setStyleSheet("color: #1C1C1E; font-size: 64px; font-weight: 900;")
        
        m_layout.addWidget(rep_title)
        m_layout.addWidget(self.rep_label)
        right_vbox.addWidget(self.metrics_card)

        self.feedback_card = QFrame()
        self.feedback_card.setFixedHeight(180)
        self.feedback_card.setStyleSheet("background: #007AFF; border-radius: 25px;")
        
        fb_layout = QVBoxLayout(self.feedback_card)
        fb_title = QLabel("CLINICAL FEEDBACK")
        fb_title.setStyleSheet("color: rgba(255,255,255,0.8); font-size: 12px; font-weight: 800;")
        
        self.ui_feedback_text = QLabel(self.feedback_text)
        self.ui_feedback_text.setWordWrap(True)
        self.ui_feedback_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.ui_feedback_text.setStyleSheet("color: white; font-size: 22px; font-weight: 700; margin-top: 10px;")
        
        fb_layout.addWidget(fb_title)
        fb_layout.addWidget(self.ui_feedback_text)
        right_vbox.addWidget(self.feedback_card)

        right_vbox.addStretch()
        self.main_layout.addLayout(right_vbox, stretch=1)

    def combo_style(self):
        return """
            QComboBox { background-color: white; border-radius: 15px; padding: 12px 20px;
                        font-size: 16px; font-weight: 600; color: #1C1C1E; border: 1px solid #E5E5EA; }
            QComboBox::drop-down { border: none; }
        """

    def update_sub_combo(self, text):
        self.secondary_combo.clear()
        if text == "Full Body":
            self.secondary_combo.addItems(["Squat (Hips, Knees, Ankles)"])
        elif text == "Hands (Upper Body)":
            self.secondary_combo.addItems(["Elbow Flexion (Bicep Curl)"])
        elif text == "Legs (Lower Body)":
            self.secondary_combo.addItems(["Knee Extension"])

    def toggle_exercise(self):
        self.is_exercising = not self.is_exercising
        self.angle_buffer.clear() # Clear smoothing buffer on start
        
        if self.is_exercising:
            self.toggle_btn.setText("STOP EXERCISE")
            self.toggle_btn.setStyleSheet("QPushButton { background-color: #FF3B30; color: white; font-size: 16px; font-weight: 800; border-radius: 20px; padding: 18px; }")
            self.rep_count = 0
            self.rep_label.setText("0")
            self.current_stage = "down"
            self.set_feedback("Position yourself in the frame.", (255, 255, 255))
        else:
            self.toggle_btn.setText("START EXERCISE")
            self.toggle_btn.setStyleSheet("QPushButton { background-color: #34C759; color: white; font-size: 16px; font-weight: 800; border-radius: 20px; padding: 18px; }")
            self.set_feedback("Exercise stopped. Good job!", (255, 255, 255))

    def set_feedback(self, text, rgb_color):
        self.feedback_text = text
        self.ui_feedback_text.setText(text)
        
        if rgb_color == (0, 255, 0): bg = "#34C759"
        elif rgb_color == (255, 0, 0): bg = "#FF3B30"
        else: bg = "#007AFF"
        self.feedback_card.setStyleSheet(f"background: {bg}; border-radius: 25px;")

        current_time = time.time()
        if current_time - self.last_speech_time > 3.0: 
            self.tts.say(text)
            self.last_speech_time = current_time

    def are_landmarks_visible(self, landmarks, required_indices, threshold=0.6):
        """Checks if all required joints for an exercise are clearly visible to prevent false tracking."""
        for idx in required_indices:
            if landmarks[idx].visibility < threshold:
                return False
        return True

    def calculate_angle(self, a, b, c):
        """Computes and smooths the 2D angle between 3 points."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0: angle = 360 - angle
        
        # Temporal Smoothing (Low-Pass Filter)
        self.angle_buffer.append(angle)
        smoothed_angle = sum(self.angle_buffer) / len(self.angle_buffer)
        return smoothed_angle

    def draw_text_with_bg(self, img, text, pos, font_scale=0.7, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
        """Draws high-visibility text with a semi-transparent dark background."""
        font = cv.FONT_HERSHEY_DUPLEX
        thickness = 2
        (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, thickness)
        
        x, y = pos
        # Draw background rectangle
        overlay = img.copy()
        cv.rectangle(overlay, (x - 5, y + baseline), (x + text_width + 5, y - text_height - 5), bg_color, -1)
        cv.addWeighted(overlay, 0.6, img, 0.4, 0, img) # 60% opacity background
        
        # Draw text
        cv.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv.LINE_AA)

    def draw_neon_skeleton(self, img, landmarks):
        """Draws a futuristic, glowing skeleton instead of standard MediaPipe lines."""
        h, w, _ = img.shape
        # Define major connections for a clean look
        connections = [
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
            (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            (self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
            (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        ]

        # Draw glowing bones
        for connection in connections:
            start_idx, end_idx = connection[0].value, connection[1].value
            
            # Only draw if both joints are visible
            if landmarks[start_idx].visibility > 0.5 and landmarks[end_idx].visibility > 0.5:
                start_pt = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                end_pt = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                
                # Neon Glow Effect (Cyan)
                cv.line(img, start_pt, end_pt, (255, 255, 0), 6, cv.LINE_AA) # Outer glow
                cv.line(img, start_pt, end_pt, (255, 255, 255), 2, cv.LINE_AA) # Inner core
                
        # Draw glowing joints
        for lm in landmarks:
            if lm.visibility > 0.5:
                pt = (int(lm.x * w), int(lm.y * h))
                cv.circle(img, pt, 5, (0, 255, 255), -1, cv.LINE_AA) # Yellow outer
                cv.circle(img, pt, 2, (255, 255, 255), -1, cv.LINE_AA) # White inner

    def process_rehabilitation_logic(self, img, landmarks):
        if not self.is_exercising: return
        exercise = self.secondary_combo.currentText()
        h, w, _ = img.shape
        
        try:
            if exercise == "Elbow Flexion (Bicep Curl)":
                required = [self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                            self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
                            self.mp_pose.PoseLandmark.LEFT_WRIST.value]
                            
                if not self.are_landmarks_visible(landmarks, required):
                    self.draw_text_with_bg(img, "WARNING: Left Arm Occluded", (50, 50), text_color=(0,0,255))
                    return

                shoulder = [landmarks[required[0]].x * w, landmarks[required[0]].y * h]
                elbow = [landmarks[required[1]].x * w, landmarks[required[1]].y * h]
                wrist = [landmarks[required[2]].x * w, landmarks[required[2]].y * h]
                
                angle = self.calculate_angle(shoulder, elbow, wrist)
                
                if angle > 150:
                    self.current_stage = "down"
                    self.set_feedback("Good extension. Now curl upwards.", (0, 122, 255))
                    color = (255, 150, 0) # Neon Blue for waiting
                if angle < 40 and self.current_stage == "down":
                    self.current_stage = "up"
                    self.rep_count += 1
                    self.rep_label.setText(str(self.rep_count))
                    self.set_feedback("Perfect curl! Slowly lower it.", (0, 255, 0))
                    color = (0, 255, 0) # Neon Green for success
                elif 40 <= angle <= 150:
                    color = (0, 255, 255) # Yellow in motion
                    
                # AR Joint Highlight
                elbow_pt = tuple(np.multiply(elbow, 1).astype(int))
                cv.circle(img, elbow_pt, 20, color, 3, cv.LINE_AA)
                self.draw_text_with_bg(img, f"Angle: {int(angle)}/deg", (elbow_pt[0] + 30, elbow_pt[1]))

            elif exercise == "Squat (Hips, Knees, Ankles)":
                required = [self.mp_pose.PoseLandmark.LEFT_HIP.value,
                            self.mp_pose.PoseLandmark.LEFT_KNEE.value,
                            self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
                            
                if not self.are_landmarks_visible(landmarks, required):
                    self.draw_text_with_bg(img, "WARNING: Left Leg Occluded", (50, 50), text_color=(0,0,255))
                    return

                hip = [landmarks[required[0]].x * w, landmarks[required[0]].y * h]
                knee = [landmarks[required[1]].x * w, landmarks[required[1]].y * h]
                ankle = [landmarks[required[2]].x * w, landmarks[required[2]].y * h]
                
                angle = self.calculate_angle(hip, knee, ankle)
                
                if angle > 160:
                    self.current_stage = "up"
                    self.set_feedback("Keep your back straight and lower your hips.", (0, 122, 255))
                    color = (255, 150, 0)
                if angle < 90 and self.current_stage == "up":
                    self.current_stage = "down"
                    self.rep_count += 1
                    self.rep_label.setText(str(self.rep_count))
                    self.set_feedback("Great depth! Drive up through your heels.", (0, 255, 0))
                    color = (0, 255, 0)
                elif 90 <= angle <= 160:
                    if self.current_stage == "up":
                        self.set_feedback("Go lower... aim for 90 degrees.", (255, 0, 0))
                    color = (0, 255, 255)

                knee_pt = tuple(np.multiply(knee, 1).astype(int))
                cv.circle(img, knee_pt, 25, color, 4, cv.LINE_AA)
                self.draw_text_with_bg(img, f"Depth Angle: {int(angle)}/deg", (knee_pt[0] + 35, knee_pt[1]))
                
        except Exception as e:
            pass 

    def main_loop(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        frame = cv.flip(frame, 1) 
        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        results = self.pose.process(rgb_image)
        
        if results.pose_landmarks:
            # Draw custom glowing skeleton
            self.draw_neon_skeleton(frame, results.pose_landmarks.landmark)
            # Run calculations and draw metrics
            self.process_rehabilitation_logic(frame, results.pose_landmarks.landmark)
            
        h, w, ch = frame.shape
        qt_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
        self.video_display.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 10))
    window = RehabPoseApp()
    window.show()
    sys.exit(app.exec())