import sys
import csv
import copy
import time
import itertools
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QFrame, 
                               QGraphicsDropShadowEffect, QSizePolicy, QSlider)
from PySide6.QtCore import QTimer, Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QColor, QFont, QCursor
from PySide6.QtTextToSpeech import QTextToSpeech  # Native TTS Import

# --- NLP ENGINE ---
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Global caches so we don't reload the massive AI model on every click, 
# which would freeze the application.
_nlp_tokenizer = None
_nlp_model = None

class NLPWorker(QThread):
    """
    Background thread to handle heavy NLP text generation. 
    Running this on the main thread would freeze the camera feed.
    """
    finished = Signal(str) # Signal emitted when the AI finishes thinking

    def __init__(self, text_buffer):
        super().__init__()
        self.text_buffer = text_buffer

    def run(self):
        global _nlp_tokenizer, _nlp_model
        
        # 1. Handle empty buffer (User clicked without making signs)
        if not self.text_buffer:
            self.finished.emit("NO SIGNS DETECTED YET.")
            return

        try:
            # 2. Explicitly load the T5 model and tokenizer into memory (First run only)
            if _nlp_tokenizer is None or _nlp_model is None:
                print("Loading AI Model into memory for the first time...")
                model_name = "vennify/t5-base-grammar-correction"
                _nlp_tokenizer = AutoTokenizer.from_pretrained(model_name)
                _nlp_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # 3. Process the raw keywords into a coherent sentence
            # 'gecc:' is the specific prompt trigger for this Vennify grammar model
            raw_text = "gecc: " + " ".join(self.text_buffer)
            inputs = _nlp_tokenizer(raw_text, return_tensors="pt", max_length=128, truncation=True)
            outputs = _nlp_model.generate(**inputs, max_length=64)
            
            # Decode the AI output and send it back to the UI
            corrected_text = _nlp_tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.finished.emit(corrected_text.upper())
            
        except ImportError as ie:
            error_msg = f"MISSING LIBRARY: {ie}. \nPlease run: pip install torch sentencepiece"
            print(error_msg)
            self.finished.emit(error_msg)
        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            print(f"NLP Error: {e}")
            self.finished.emit(error_msg)


# --- Logic Fallbacks ---
# Mock classifier just in case the real model isn't downloaded yet.
try:
    from model import KeyPointClassifier
except ImportError:
    class KeyPointClassifier:
        def __call__(self, x): return 0, 0.95


class SignSpeakApp(QMainWindow):
    """
    Main Application Window holding the UI, Camera Loop, and AI triggers.
    """
    def __init__(self):
        super().__init__()
        self.load_labels()
        
        # Buffer to hold the last 5 recognized words for sentence reconstruction
        self.word_buffer = deque(maxlen=5)
        self.last_detected_word = None
        
        # --- STABILIZATION TRACKER ---
        # Prevents "flickering" detections. The user must hold a sign for X seconds
        # before it is officially added to the history buffer.
        self.stabilization_threshold = 1.0 
        self.current_proposed_word = None
        self.proposed_word_start_time = 0

        # --- INIT NATIVE TTS ENGINE ---
        # Used to speak the final translated sentence out loud
        self.tts = QTextToSpeech(self)

        # Initialize MediaPipe for Hand Tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.keypoint_classifier = KeyPointClassifier()

        # Build the user interface
        self.init_ui()
        
        # Start the Camera and Timer (Main Loop runs every 30ms / ~33 FPS)
        self.cap = cv.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.main_loop)
        self.timer.start(30)

    def load_labels(self):
        """Loads the gesture vocabulary from a CSV file."""
        try:
            with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
                self.keypoint_labels = [row[0].upper() for row in csv.reader(f)]
        except:
            self.keypoint_labels = ["HELLO", "ME", "GO", "STORE", "HELP"]

    def init_ui(self):
        """Constructs the PySide6 Graphical User Interface."""
        self.setWindowTitle("SignSpeak AI | NLP Integrated")
        self.setMinimumSize(1250, 850)
        self.setStyleSheet("background-color: #F2F2F7;")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setContentsMargins(40, 40, 40, 40)
        self.main_layout.setSpacing(40)

        # --- LEFT SECTION: CAMERA & CONTROLS ---
        left_container = QWidget()
        left_vbox = QVBoxLayout(left_container)
        left_vbox.setSpacing(20)
        
        title = QLabel("SignSpeak")
        title.setStyleSheet("font-size: 38px; font-weight: 800; color: #1C1C1E; letter-spacing: -1.5px;")
        left_vbox.addWidget(title)

        # Video Display Frame
        self.video_frame = QFrame()
        self.video_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_frame.setStyleSheet("background-color: #1C1C1E; border-radius: 30px;")
        
        v_layout = QVBoxLayout(self.video_frame)
        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignCenter)
        v_layout.addWidget(self.video_display)
        
        shadow = QGraphicsDropShadowEffect(blurRadius=60, xOffset=0, yOffset=25, color=QColor(0,0,0,40))
        self.video_frame.setGraphicsEffect(shadow)
        left_vbox.addWidget(self.video_frame, stretch=1)

        # Control Bar (Buttons at the bottom)
        self.control_bar = QFrame()
        self.control_bar.setFixedHeight(85)
        self.control_bar.setStyleSheet("background-color: white; border-radius: 25px;")
        
        cb_shadow = QGraphicsDropShadowEffect(blurRadius=20, xOffset=0, yOffset=10, color=QColor(0,0,0,15))
        self.control_bar.setGraphicsEffect(cb_shadow)
        
        cb_l = QHBoxLayout(self.control_bar)
        cb_l.setContentsMargins(25, 10, 25, 10)
        
        # Button: Reconstruct Sentence
        self.nlp_btn = QPushButton("✨ RECONSTRUCT SENTENCE")
        self.nlp_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.nlp_btn.setStyleSheet("""
            QPushButton {
                background-color: #007AFF; color: white; font-size: 14px;
                font-weight: 800; border-radius: 20px; padding: 12px 25px;
            }
            QPushButton:hover { background-color: #0056b3; }
            QPushButton:pressed { background-color: #003d82; }
            QPushButton:disabled { background-color: #A0C9F0; }
        """)
        self.nlp_btn.clicked.connect(self.run_nlp_reconstruction)
        cb_l.addWidget(self.nlp_btn)

        # Button: Clear History
        self.clear_btn = QPushButton("CLEAR")
        self.clear_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #000000; color: white; font-size: 14px;
                font-weight: 800; border-radius: 20px; padding: 12px 25px; margin-left: 10px;
            }
            QPushButton:hover { background-color: #D32F2F; }
            QPushButton:pressed { background-color: #B71C1C; }
        """)
        self.clear_btn.clicked.connect(self.clear_history)
        cb_l.addWidget(self.clear_btn)

        # Button: Text-to-Speech
        self.speech_btn = QPushButton("🔊 SPEECH")
        self.speech_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.speech_btn.setStyleSheet("""
            QPushButton {
                background-color: #AF52DE; color: white; font-size: 14px;
                font-weight: 800; border-radius: 20px; padding: 12px 25px; margin-left: 10px;
            }
            QPushButton:hover { background-color: #8E3BB8; }
            QPushButton:pressed { background-color: #6C2B8C; }
        """)
        self.speech_btn.clicked.connect(self.speak_text)
        cb_l.addWidget(self.speech_btn)
        
        cb_l.addStretch()
        
        # Engine Status Indicator
        self.status_label = QLabel("● AI ENGINE READY")
        self.status_label.setStyleSheet("color: #34C759; font-weight: 800; font-size: 13px; letter-spacing: 1px;")
        cb_l.addWidget(self.status_label)
        left_vbox.addWidget(self.control_bar)

        # --- RIGHT SECTION: DATA & SETTINGS ---
        right_container = QWidget()
        right_container.setFixedWidth(400)
        right_vbox = QVBoxLayout(right_container)
        right_vbox.setSpacing(20)

        # NLP Interpretation Card (Displays the translated sentence)
        self.nlp_card = QFrame()
        self.nlp_card.setFixedHeight(220)
        self.nlp_card.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #007AFF, stop:1 #0051FF);
                border-radius: 30px;
            }
        """)
        nlp_shadow = QGraphicsDropShadowEffect(blurRadius=40, xOffset=0, yOffset=15, color=QColor(0, 122, 255, 80))
        self.nlp_card.setGraphicsEffect(nlp_shadow)
        
        nlp_l = QVBoxLayout(self.nlp_card)
        nlp_l.setContentsMargins(30, 30, 30, 30)
        
        nlp_tag = QLabel("AI INTERPRETATION")
        nlp_tag.setStyleSheet("color: rgba(255,255,255,0.8); font-size: 12px; font-weight: 800; background: transparent;")
        self.nlp_text = QLabel("Waiting for input...")
        self.nlp_text.setWordWrap(True)
        self.nlp_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.nlp_text.setStyleSheet("color: white; font-size: 20px; font-weight: 700; background: transparent; margin-top: 10px;")
        
        nlp_l.addWidget(nlp_tag)
        nlp_l.addWidget(self.nlp_text)
        right_vbox.addWidget(self.nlp_card)

        # Slider: Lock-in Timing Adjustment
        slider_layout = QVBoxLayout()
        slider_header = QHBoxLayout()
        sl_label = QLabel("Lock-in Time:")
        sl_label.setStyleSheet("font-size: 15px; font-weight: 700; color: #8E8E93;")
        self.sl_val_label = QLabel("1.0s")
        self.sl_val_label.setStyleSheet("font-size: 15px; font-weight: 800; color: #007AFF;")
        
        slider_header.addWidget(sl_label)
        slider_header.addStretch()
        slider_header.addWidget(self.sl_val_label)
        
        self.delay_slider = QSlider(Qt.Horizontal)
        self.delay_slider.setMinimum(1)   
        self.delay_slider.setMaximum(100) 
        self.delay_slider.setValue(10)    
        self.delay_slider.setStyleSheet("""
            QSlider::groove:horizontal { border-radius: 4px; height: 8px; background: #E5E5EA; }
            QSlider::handle:horizontal { background: #007AFF; width: 18px; margin: -5px 0; border-radius: 9px; }
        """)
        self.delay_slider.valueChanged.connect(self.on_slider_change)
        
        slider_layout.addLayout(slider_header)
        slider_layout.addWidget(self.delay_slider)
        right_vbox.addLayout(slider_layout)

        # Detection History Panel
        history_header = QLabel("Detection History")
        history_header.setStyleSheet("font-size: 18px; font-weight: 700; color: #8E8E93; margin-top: 5px;")
        right_vbox.addWidget(history_header)

        # Create 3 History Cards programmatically
        self.history_cards = []
        for i in range(3):
            is_latest = (i == 0) # Highlight the top card
            card = self.create_ios_card(is_latest)
            self.history_cards.append(card)
            right_vbox.addWidget(card)
        
        right_vbox.addStretch()
        self.main_layout.addWidget(left_container, stretch=3)
        self.main_layout.addWidget(right_container, stretch=1)

    # --- ACTION METHODS ---

    def speak_text(self):
        """Triggers the OS text-to-speech engine to read the NLP output."""
        text_to_read = self.nlp_text.text()
        if text_to_read and text_to_read != "Waiting for input..." and "ERROR" not in text_to_read:
            self.tts.say(text_to_read)

    def clear_history(self):
        """Resets all backend memory buffers and clears the UI history cards."""
        self.word_buffer.clear()
        self.last_detected_word = None
        self.current_proposed_word = None
        self.proposed_word_start_time = time.time()
        
        self.nlp_text.setText("Waiting for input...")
        for card in self.history_cards:
            card.text_widget.setText("---")

    def on_slider_change(self, value):
        """Updates the stabilization timer based on slider input."""
        self.stabilization_threshold = value / 10.0
        self.sl_val_label.setText(f"{self.stabilization_threshold:.1f}s")

    def create_ios_card(self, is_latest=False):
        """Helper to create visually appealing UI cards for the detection history."""
        card = QFrame()
        card_shadow = QGraphicsDropShadowEffect(blurRadius=20, xOffset=0, yOffset=8, color=QColor(0,0,0,10))
        card.setGraphicsEffect(card_shadow)
        
        if is_latest:
            card.setFixedHeight(110)
            card.setStyleSheet("background-color: #E5F1FF; border: 2px solid #007AFF; border-radius: 25px;")
            tag_text = "CURRENT SIGN"
            tag_color = "#007AFF"
            txt_color = "#007AFF"
            txt_size = "28px"
        else:
            card.setFixedHeight(90)
            card.setStyleSheet("background-color: white; border: none; border-radius: 25px;")
            tag_text = "PREVIOUS SIGN"
            tag_color = "#8E8E93"
            txt_color = "#1C1C1E"
            txt_size = "22px"

        v = QVBoxLayout(card)
        v.setContentsMargins(25, 15, 25, 15)
        v.setAlignment(Qt.AlignVCenter)
        
        tag = QLabel(tag_text)
        tag.setStyleSheet(f"font-size: 10px; font-weight: 800; color: {tag_color}; border: none; background: transparent;")
        
        txt = QLabel("---")
        txt.setStyleSheet(f"font-size: {txt_size}; font-weight: 800; color: {txt_color}; border: none; background: transparent;")
        
        v.addWidget(tag)
        v.addWidget(txt)
        card.text_widget = txt
        return card

    # --- CORE COMPUTER VISION LOOP ---

    def main_loop(self):
        """
        Runs every 30ms. Reads the camera, detects hands, classifies the gesture, 
        and updates the holographic UI overlay.
        """
        ret, frame = self.cap.read()
        if not ret: return
        
        # Mirror image for natural user experience
        frame = cv.flip(frame, 1)
        debug_image = copy.deepcopy(frame) # Used to draw the AR holograms
        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # MediaPipe requires RGB
        
        # Detect hands
        results = self.hands.process(rgb_image)

        if results.multi_hand_landmarks:
            hand_data = []
            for hand_landmarks in results.multi_hand_landmarks:
                # 1. Extract raw coordinates
                lp = self.calc_landmark_list(debug_image, hand_landmarks)
                # 2. Normalize relative to the wrist
                pre_processed = self.pre_process_landmark(lp)
                
                # 3. Predict the gesture using your custom model
                prediction = self.keypoint_classifier(pre_processed)
                if isinstance(prediction, (tuple, list)):
                    idx, confidence = prediction[0], float(prediction[1])
                else:
                    idx, confidence = prediction, 0.0
                    
                word = self.keypoint_labels[idx]
                hand_data.append({"lp": lp, "word": word, "conf": confidence})

            # For stabilization, we only look at the first detected hand
            current_frame_word = hand_data[0]["word"]

            # --- STABILIZATION LOGIC ---
            # If the user is holding the SAME sign as the previous frame
            if current_frame_word == self.current_proposed_word:
                elapsed = time.time() - self.proposed_word_start_time
                hold_progress = min(1.0, elapsed / self.stabilization_threshold)
                
                # If they held it long enough (surpassed the threshold)
                if elapsed >= self.stabilization_threshold:
                    # And it's not just a repeat of the word already logged
                    if current_frame_word != self.last_detected_word:
                        self.word_buffer.append(current_frame_word)
                        self.last_detected_word = current_frame_word
                        self.update_history_ui(current_frame_word)
            else:
                # The sign changed. Reset the timer and tracking.
                self.current_proposed_word = current_frame_word
                self.proposed_word_start_time = time.time()
                hold_progress = 0.0

            # Draw holographic UI components
            for data in hand_data:
                self.draw_holographic_ar(debug_image, data["lp"])
                hp = hold_progress if data["word"] == self.current_proposed_word else 0.0
                self.draw_confidence_slider(debug_image, data["lp"], data["conf"], data["word"], hp)
        else:
            # No hands detected, reset timers
            self.current_proposed_word = None
            self.proposed_word_start_time = time.time()

        # Convert the OpenCV image to a PySide6 Pixmap to display in the UI
        h, w, ch = debug_image.shape
        qt_img = QImage(debug_image.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
        self.video_display.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # --- NLP INTEGRATION ---

    def run_nlp_reconstruction(self):
        """Disables buttons and spawns the background NLP thread."""
        self.status_label.setText("● AI THINKING...")
        self.status_label.setStyleSheet("color: #FF9F0A; font-weight: 800; font-size: 13px; letter-spacing: 1px;")
        
        # Prevent user interaction while processing
        self.nlp_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.speech_btn.setEnabled(False) 
        
        # Pass the current memory buffer to the AI
        self.nlp_worker = NLPWorker(list(self.word_buffer))
        self.nlp_worker.finished.connect(self.on_nlp_finished)
        self.nlp_worker.start()

    def on_nlp_finished(self, text):
        """Callback for when the NLP thread finishes processing."""
        self.nlp_text.setText(text)
        self.status_label.setText("● AI ENGINE READY")
        self.status_label.setStyleSheet("color: #34C759; font-weight: 800; font-size: 13px; letter-spacing: 1px;")
        
        # Re-enable interaction
        self.nlp_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.speech_btn.setEnabled(True) 

    # --- DRAWING / VISUALS ---

    def draw_holographic_ar(self, img, lp):
        """Draws the futuristic skeletal connections between MediaPipe hand landmarks."""
        # Define connections based on human hand anatomy (Thumb, Index, Middle, Ring, Pinky)
        paths = [(0,1,2,3,4), (0,5,6,7,8), (0,9,10,11,12), (0,13,14,15,16), (0,17,18,19,20)]
        for path in paths:
            for i in range(len(path)-1):
                cv.line(img, tuple(lp[path[i]]), tuple(lp[path[i+1]]), (255, 122, 0), 2, cv.LINE_AA)
        
        # Draw the joints
        for pt in lp:
            cv.circle(img, tuple(pt), 5, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(img, tuple(pt), 7, (255, 122, 0), 1, cv.LINE_AA)

    def draw_confidence_slider(self, img, lp, confidence, word, hold_progress=0.0):
        """Draws the floating progress bar/timer underneath the hand in real-time."""
        x_coords = [pt[0] for pt in lp]
        y_coords = [pt[1] for pt in lp]
        min_x, max_x = min(x_coords), max(x_coords)
        max_y = max(y_coords)
        
        bar_width = max(140, max_x - min_x) 
        start_x = min_x + (max_x - min_x) // 2 - bar_width // 2
        start_y = max_y + 40 
        
        # Determine color based on ML confidence score
        if confidence < 0.5: color = (60, 60, 255)       # Red
        elif confidence < 0.8: color = (0, 165, 255)     # Orange
        else: color = (50, 205, 50)                      # Green
            
        # Draw translucent background box
        overlay = img.copy()
        padding = 15
        cv.rectangle(overlay, (start_x - padding, start_y - 30), 
                     (start_x + bar_width + padding, start_y + 22), 
                     (28, 28, 30), -1)
        cv.addWeighted(overlay, 0.65, img, 0.35, 0, img)
        
        # Draw Confidence Bar
        cv.line(img, (start_x, start_y), (start_x + bar_width, start_y), (80, 80, 80), 6, cv.LINE_AA)
        fill_width = int(bar_width * confidence)
        if fill_width > 0:
            cv.line(img, (start_x, start_y), (start_x + fill_width, start_y), color, 6, cv.LINE_AA)
            
        # Draw Stabilization Timer Bar underneath
        timer_y = start_y + 10
        cv.line(img, (start_x, timer_y), (start_x + bar_width, timer_y), (80, 80, 80), 3, cv.LINE_AA)
        timer_width = int(bar_width * hold_progress)
        if timer_width > 0:
            timer_color = (0, 200, 255) if hold_progress < 1.0 else (50, 205, 50)
            cv.line(img, (start_x, timer_y), (start_x + timer_width, timer_y), timer_color, 3, cv.LINE_AA)
            
        # Draw Word Text
        text = f"{word} ({int(confidence * 100)}%)"
        font = cv.FONT_HERSHEY_DUPLEX
        font_scale = 0.45
        text_size = cv.getTextSize(text, font, font_scale, 1)[0]
        text_x = start_x + (bar_width - text_size[0]) // 2
        cv.putText(img, text, (text_x, start_y - 12), font, font_scale, (255, 255, 255), 1, cv.LINE_AA)

    def update_history_ui(self, word):
        """Shifts older words down and places the newly locked word at the top."""
        self.history_cards[2].text_widget.setText(self.history_cards[1].text_widget.text())
        self.history_cards[1].text_widget.setText(self.history_cards[0].text_widget.text())
        self.history_cards[0].text_widget.setText(word)

    # --- MATH & NORMALIZATION ---

    def calc_landmark_list(self, img, landmarks):
        """Converts normalized MediaPipe values (0.0 - 1.0) into actual pixel dimensions."""
        w, h = img.shape[1], img.shape[0]
        return [[int(lm.x * w), int(lm.y * h)] for lm in landmarks.landmark]

    def pre_process_landmark(self, lp):
        """
        Normalizes the hand coordinates so the machine learning model can recognize 
        the shape regardless of where it is on the screen or how close the hand is to the camera.
        """
        temp = copy.deepcopy(lp)
        
        # Treat the wrist (point 0) as the origin (0,0)
        bx, by = temp[0][0], temp[0][1]
        for i in range(len(temp)):
            temp[i][0] -= bx
            temp[i][1] -= by
            
        # Flatten the 2D array into a 1D array
        temp = list(itertools.chain.from_iterable(temp))
        
        # Scale all values between -1 and 1
        max_v = max(map(abs, temp)) if temp else 1
        return [n / max_v for n in temp]


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setFont(QFont(".AppleSystemUIFont", 10))
    window = SignSpeakApp()
    window.show()
    sys.exit(app.exec())