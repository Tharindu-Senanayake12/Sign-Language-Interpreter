import sys
import csv
import copy
import time
import itertools
import os
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QFrame, 
                               QGraphicsDropShadowEffect, QSizePolicy, QSlider,
                               QStackedWidget)
from PySide6.QtCore import QTimer, Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QColor, QFont, QCursor
from PySide6.QtTextToSpeech import QTextToSpeech  

# --- NLP ENGINE ---
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

_nlp_tokenizer = None
_nlp_model = None

class NLPWorker(QThread):
    finished = Signal(str)

    def __init__(self, text_buffer):
        super().__init__()
        self.text_buffer = text_buffer

    def run(self):
        global _nlp_tokenizer, _nlp_model
        
        if not self.text_buffer:
            self.finished.emit("NO SIGNS DETECTED YET.")
            return

        try:
            if _nlp_tokenizer is None or _nlp_model is None:
                print("Loading AI Model into memory for the first time...")
                model_name = "vennify/t5-base-grammar-correction"
                _nlp_tokenizer = AutoTokenizer.from_pretrained(model_name)
                _nlp_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            raw_text = "gecc: " + " ".join(self.text_buffer)
            inputs = _nlp_tokenizer(raw_text, return_tensors="pt", max_length=128, truncation=True)
            outputs = _nlp_model.generate(**inputs, max_length=64)
            
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
try:
    from model import KeyPointClassifier
except ImportError:
    class KeyPointClassifier:
        def __call__(self, x): return 0, 0.95

try:
    from model import PointHistoryClassifier
except ImportError:
    class PointHistoryClassifier:
        def __call__(self, x): return 0, 0.0

# _____________Main Application Window________________

class SignSpeakApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.load_labels()
        
        # Buffer and tracking
        self.word_buffer = deque(maxlen=5)
        self.last_detected_word = None
        self.stabilization_threshold = 1.0 
        self.current_proposed_word = None
        self.proposed_word_start_time = 0

        # --- DATA COLLECTION & LOGGING STATE ---
        self.in_logging_ui_mode = False
        self.mode = 0  # 0: Normal, 1: Keypoint Logging, 2: Point History Logging
        self.label_buffer = ""
        self.logging_label = -1
        self.capture_next = False

        # --- DYNAMIC GESTURE TRACKING ---
        self.motion_pattern_enabled = False
        self.point_history = deque(maxlen=16) 

        self.tts = QTextToSpeech(self)

        # Initialize MediaPipe for Hand Tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()

        self.init_ui()
        
        self.cap = cv.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.main_loop)
        self.timer.start(30)

    def load_labels(self):
        try:
            with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
                self.keypoint_labels = [row[0].upper() for row in csv.reader(f)]
        except:
            self.keypoint_labels = ["HELLO", "ME", "GO", "STORE", "HELP"]

        try:
            with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
                self.point_history_labels = [row[0].upper() for row in csv.reader(f)]
        except:
            self.point_history_labels = ["CLOCKWISE", "COUNTER-CLOCKWISE", "SWIPE-RIGHT"]

    def init_ui(self):
        self.setWindowTitle("SignSpeak AI | NLP Integrated")
        self.setMinimumSize(1300, 850)
        self.setStyleSheet("background-color: #F2F2F7;")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 1. Main Layout is now a Vertical Box (Header on top, Body below)
        self.main_layout = QVBoxLayout(central_widget)
        self.main_layout.setContentsMargins(40, 40, 40, 40)
        self.main_layout.setSpacing(20)

        # --- GLOBAL HEADER SECTION ---
        header_layout = QHBoxLayout()
        title = QLabel("SignSpeak")
        title.setStyleSheet("font-size: 38px; font-weight: 800; color: #1C1C1E; letter-spacing: -1.5px;")
        
        self.toggle_log_mode_btn = QPushButton("ENTER LOGGING MODE")
        self.toggle_log_mode_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.toggle_log_mode_btn.setStyleSheet("""
            QPushButton { background-color: #FF3B30; color: white; font-size: 13px; font-weight: 800; border-radius: 15px; padding: 10px 20px; }
            QPushButton:hover { background-color: #D32F2F; }
        """)
        self.toggle_log_mode_btn.clicked.connect(self.toggle_logging_ui)

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.toggle_log_mode_btn)
        
        self.main_layout.addLayout(header_layout)

        # --- BODY SECTION (Camera Left, Panels Right) ---
        body_layout = QHBoxLayout()
        body_layout.setSpacing(40)

        # --- LEFT SECTION ---
        left_container = QWidget()
        left_vbox = QVBoxLayout(left_container)
        left_vbox.setContentsMargins(0, 0, 0, 0) # Remove margins so it aligns perfectly
        left_vbox.setSpacing(20)

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

        self.control_bar = QFrame()
        self.control_bar.setFixedHeight(85)
        self.control_bar.setStyleSheet("background-color: white; border-radius: 25px;")
        cb_shadow = QGraphicsDropShadowEffect(blurRadius=20, xOffset=0, yOffset=10, color=QColor(0,0,0,15))
        self.control_bar.setGraphicsEffect(cb_shadow)
        
        cb_l = QHBoxLayout(self.control_bar)
        cb_l.setContentsMargins(25, 10, 25, 10)
    
        self.nlp_btn = QPushButton("RECONSTRUCT SENTENCE")
        self.nlp_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.nlp_btn.setStyleSheet("""
            QPushButton { background-color: #007AFF; color: white; font-size: 14px; font-weight: 800; border-radius: 20px; padding: 12px 25px; }
            QPushButton:hover { background-color: #0056b3; }
            QPushButton:disabled { background-color: #A0C9F0; }
        """)
        self.nlp_btn.clicked.connect(self.run_nlp_reconstruction)
        cb_l.addWidget(self.nlp_btn)

        self.clear_btn = QPushButton("CLEAR")
        self.clear_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.clear_btn.setStyleSheet("""
            QPushButton { background-color: #000000; color: white; font-size: 14px; font-weight: 800; border-radius: 20px; padding: 12px 25px; margin-left: 10px; }
            QPushButton:hover { background-color: #333333; }
        """)
        self.clear_btn.clicked.connect(self.clear_history)
        cb_l.addWidget(self.clear_btn)

        self.speech_btn = QPushButton("🔊 SPEECH")
        self.speech_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.speech_btn.setStyleSheet("""
            QPushButton { background-color: #AF52DE; color: white; font-size: 14px; font-weight: 800; border-radius: 20px; padding: 12px 25px; margin-left: 10px; }
            QPushButton:hover { background-color: #8E3BB8; }
        """)
        self.speech_btn.clicked.connect(self.speak_text)
        cb_l.addWidget(self.speech_btn)

        self.motion_btn = QPushButton("MOTION PATTERNS: OFF")
        self.motion_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.motion_btn.setStyleSheet("""
            QPushButton { background-color: #E5E5EA; color: #8E8E93; font-size: 13px; font-weight: 800; border-radius: 20px; padding: 12px 20px; margin-left: 10px; border: 2px solid #D1D1D6;}
            QPushButton:hover { background-color: #D1D1D6; }
        """)
        self.motion_btn.clicked.connect(self.toggle_motion_patterns)
        cb_l.addWidget(self.motion_btn)
        
        cb_l.addStretch()
        
        self.status_label = QLabel("● AI READY")
        self.status_label.setStyleSheet("color: #34C759; font-weight: 800; font-size: 13px; letter-spacing: 1px;")
        cb_l.addWidget(self.status_label)
        left_vbox.addWidget(self.control_bar)

        # --- RIGHT SECTION (QStackedWidget) ---
        self.right_stack = QStackedWidget()
        self.right_stack.setFixedWidth(400)

        # --- PAGE 0: Normal Mode ---
        self.page_normal = QWidget()
        right_vbox = QVBoxLayout(self.page_normal)
        right_vbox.setContentsMargins(0, 0, 0, 0)
        right_vbox.setSpacing(20)

        self.nlp_card = QFrame()
        self.nlp_card.setFixedHeight(220)
        self.nlp_card.setStyleSheet("QFrame { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #007AFF, stop:1 #0051FF); border-radius: 30px; }")
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

        history_header = QLabel("Detection History")
        history_header.setStyleSheet("font-size: 18px; font-weight: 700; color: #8E8E93; margin-top: 5px;")
        right_vbox.addWidget(history_header)

        self.history_cards = []
        for i in range(3):
            card = self.create_ios_card(is_latest=(i == 0))
            self.history_cards.append(card)
            right_vbox.addWidget(card)
        
        right_vbox.addStretch()

        # --- PAGE 1: Enhanced Logging Mode ---
        self.page_logging = QWidget()
        log_vbox = QVBoxLayout(self.page_logging)
        log_vbox.setContentsMargins(0, 0, 0, 0)
        log_vbox.setSpacing(20)

        # 1. Status Dashboard
        status_card = QFrame()
        status_card.setStyleSheet("background-color: #1C1C1E; border-radius: 25px;")
        status_shadow = QGraphicsDropShadowEffect(blurRadius=20, xOffset=0, yOffset=10, color=QColor(0,0,0,30))
        status_card.setGraphicsEffect(status_shadow)
        
        status_l = QVBoxLayout(status_card)
        status_l.setContentsMargins(25, 25, 25, 25)
        
        self.log_status_title = QLabel("RECORDING STANDBY")
        self.log_status_title.setStyleSheet("color: #FF453A; font-size: 13px; font-weight: 800; letter-spacing: 1px;")
        self.log_status_display = QLabel("Mode: STATIC KEYPOINT\nLabel ID: NONE")
        self.log_status_display.setStyleSheet("color: white; font-size: 20px; font-weight: 700; line-height: 1.5;")
        
        status_l.addWidget(self.log_status_title)
        status_l.addWidget(self.log_status_display)

        # 2. Key Bindings Card
        keys_card = QFrame()
        keys_card.setStyleSheet("background-color: white; border-radius: 25px;")
        keys_shadow = QGraphicsDropShadowEffect(blurRadius=20, xOffset=0, yOffset=10, color=QColor(0,0,0,15))
        keys_card.setGraphicsEffect(keys_shadow)
        
        keys_l = QVBoxLayout(keys_card)
        keys_l.setContentsMargins(25, 25, 25, 25)
        keys_l.setSpacing(18)
        
        keys_title = QLabel("KEY MAPPINGS")
        keys_title.setStyleSheet("color: #8E8E93; font-size: 12px; font-weight: 800; letter-spacing: 1px;")
        keys_l.addWidget(keys_title)

        def create_key_row(key_text, desc_text, text_color, bg_color):
            row = QHBoxLayout()
            key_lbl = QLabel(key_text)
            key_lbl.setAlignment(Qt.AlignCenter)
            key_lbl.setFixedSize(45, 40)
            key_lbl.setStyleSheet(f"background-color: {bg_color}; color: {text_color}; font-weight: 800; font-size: 15px; border-radius: 10px;")
            desc_lbl = QLabel(desc_text)
            desc_lbl.setStyleSheet("color: #1C1C1E; font-size: 15px; font-weight: 700;")
            row.addWidget(key_lbl)
            row.addWidget(desc_lbl)
            row.addStretch()
            return row

        keys_l.addLayout(create_key_row("K", "Static Mode", "#007AFF", "#E5F1FF"))
        keys_l.addLayout(create_key_row("N", "Dynamic Mode", "#FF9F0A", "#FFF4E5"))
        keys_l.addLayout(create_key_row("0-99", "Set Label ID", "#8E8E93", "#F2F2F7"))
        keys_l.addLayout(create_key_row("P", "Capture Frame", "#1A582A", "#EBF9EE"))
        keys_l.addLayout(create_key_row("ESC", "Exit App", "#FF3B30", "#FFEBEA"))

        # 3. Capture Button
        self.ui_capture_btn = QPushButton("CAPTURE")
        self.ui_capture_btn.setFixedHeight(65)
        self.ui_capture_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.ui_capture_btn.setStyleSheet("""
            QPushButton { background-color: #007AFF; color: white; font-size: 16px; font-weight: 800; border-radius: 25px; }
            QPushButton:hover { background-color: #28A745; }
            QPushButton:pressed { background-color: #1E7E34; }
        """)
        self.ui_capture_btn.clicked.connect(self.trigger_capture)

        log_vbox.addWidget(status_card)
        log_vbox.addWidget(keys_card)
        log_vbox.addStretch()
        log_vbox.addWidget(self.ui_capture_btn)

        # Add pages to stack
        self.right_stack.addWidget(self.page_normal)
        self.right_stack.addWidget(self.page_logging)

        # --- ASSEMBLE BODY ---
        body_layout.addWidget(left_container, stretch=3)
        body_layout.addWidget(self.right_stack, stretch=1)

        self.main_layout.addLayout(body_layout)

    # --- UI TOGGLE ACTIONS ---
    def toggle_logging_ui(self):
        self.in_logging_ui_mode = not self.in_logging_ui_mode
        if self.in_logging_ui_mode:
            self.right_stack.setCurrentIndex(1)
            self.toggle_log_mode_btn.setText("EXIT LOGGING MODE")
            self.toggle_log_mode_btn.setStyleSheet("QPushButton { background-color: #8E8E93; color: white; font-size: 13px; font-weight: 800; border-radius: 15px; padding: 10px 20px; }")
            self.nlp_btn.hide()
            self.speech_btn.hide()
            self.motion_btn.hide()
            self.mode = 1 
        else:
            self.right_stack.setCurrentIndex(0)
            self.toggle_log_mode_btn.setText("ENTER LOGGING MODE")
            self.toggle_log_mode_btn.setStyleSheet("QPushButton { background-color: #FF3B30; color: white; font-size: 13px; font-weight: 800; border-radius: 15px; padding: 10px 20px; } QPushButton:hover { background-color: #D32F2F; }")
            self.nlp_btn.show()
            self.speech_btn.show()
            self.motion_btn.show()
            self.mode = 0 
        self.update_log_ui_text()

    def toggle_motion_patterns(self):
        self.motion_pattern_enabled = not self.motion_pattern_enabled
        if self.motion_pattern_enabled:
            self.motion_btn.setText("MOTION PATTERNS: ON")
            self.motion_btn.setStyleSheet("""
                QPushButton { background-color: #E5F1FF; color: #007AFF; font-size: 13px; font-weight: 800; border-radius: 20px; padding: 12px 20px; margin-left: 10px; border: 2px solid #007AFF;}
                QPushButton:hover { background-color: #CCE4FF; }
            """)
        else:
            self.motion_btn.setText("MOTION PATTERNS: OFF")
            self.motion_btn.setStyleSheet("""
                QPushButton { background-color: #E5E5EA; color: #8E8E93; font-size: 13px; font-weight: 800; border-radius: 20px; padding: 12px 20px; margin-left: 10px; border: 2px solid #D1D1D6;}
                QPushButton:hover { background-color: #D1D1D6; }
            """)
            self.point_history.clear()

    # --- KEYBOARD EVENT HANDLING FOR LOGGING ---
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif self.in_logging_ui_mode:
            if key == Qt.Key_K:
                self.mode = 1
            elif key == Qt.Key_N:
                self.mode = 2
            elif Qt.Key_0 <= key <= Qt.Key_9:
                if len(self.label_buffer) < 2:
                    self.label_buffer += chr(key)
            elif key == Qt.Key_Return or key == Qt.Key_Enter:
                if self.label_buffer:
                    self.logging_label = int(self.label_buffer)
                    self.label_buffer = "" 
            elif key == Qt.Key_Backspace:
                self.label_buffer = self.label_buffer[:-1]
            elif key == Qt.Key_P:
                self.trigger_capture()
            self.update_log_ui_text()

    def trigger_capture(self):
        self.capture_next = True

    def update_log_ui_text(self):
        m_str = "STATIC KEYPOINT" if self.mode == 1 else "DYNAMIC HISTORY" if self.mode == 2 else "NORMAL"
        lbl_str = str(self.logging_label) if self.logging_label != -1 else "NONE"
        buf_str = f" (Typing: {self.label_buffer})" if self.label_buffer else ""
        self.log_status_display.setText(f"Mode: {m_str}\nLabel ID: {lbl_str}{buf_str}")
        
        if self.logging_label != -1:
            self.log_status_title.setText("● READY TO CAPTURE")
            self.log_status_title.setStyleSheet("color: #34C759; font-size: 13px; font-weight: 800; letter-spacing: 1px;")
        else:
            self.log_status_title.setText("RECORDING STANDBY")
            self.log_status_title.setStyleSheet("color: #FF453A; font-size: 13px; font-weight: 800; letter-spacing: 1px;")

    def log_keypoint_data(self, label, landmark_list):
        os.makedirs('model/keypoint_classifier', exist_ok=True)
        with open('model/keypoint_classifier/keypoint.csv', 'a', newline="") as f:
            csv.writer(f).writerow([label, *landmark_list])
        print(f"✅ Keypoint saved for label: {label}")

    def log_point_history_data(self, label, point_history_list):
        os.makedirs('model/point_history_classifier', exist_ok=True)
        with open('model/point_history_classifier/point_history.csv', 'a', newline="") as f:
            csv.writer(f).writerow([label, *point_history_list])
        print(f"✅ Point history saved for label: {label}")

    # --- CORE COMPUTER VISION LOOP ---
    def main_loop(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        frame = cv.flip(frame, 1)
        debug_image = copy.deepcopy(frame) 
        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB) 
        
        if self.in_logging_ui_mode:
            color = (0, 255, 0) if self.mode == 1 else (0, 165, 255)
            hud = f"LOGGING: {'KEYPOINT' if self.mode == 1 else 'HISTORY'} | ID: {self.logging_label}"
            cv.putText(debug_image, hud, (20, 40), cv.FONT_HERSHEY_DUPLEX, 0.7, color, 2, cv.LINE_AA)
            if self.capture_next:
                cv.circle(debug_image, (40, 80), 12, (0, 0, 255), -1) 

        results = self.hands.process(rgb_image)

        if results.multi_hand_landmarks:
            hand_data = []
            for hand_landmarks in results.multi_hand_landmarks:
                lp = self.calc_landmark_list(debug_image, hand_landmarks)
                pre_processed = self.pre_process_landmark(lp)
                
                if self.motion_pattern_enabled or self.mode == 2:
                    self.point_history.append(lp[8])
                
                pre_processed_history = self.pre_process_point_history(debug_image, self.point_history)

                if self.capture_next and self.logging_label != -1:
                    if self.mode == 1:
                        self.log_keypoint_data(self.logging_label, pre_processed)
                    elif self.mode == 2:
                        self.log_point_history_data(self.logging_label, pre_processed_history)
                    self.capture_next = False 
                
                prediction = self.keypoint_classifier(pre_processed)
                idx, confidence = prediction if isinstance(prediction, (tuple, list)) else (prediction, 0.0)
                word = self.keypoint_labels[idx] if idx < len(self.keypoint_labels) else "???"

                motion_word = ""
                if self.motion_pattern_enabled and len(self.point_history) == 16:
                    motion_pred = self.point_history_classifier(pre_processed_history)
                    m_idx, m_conf = motion_pred if isinstance(motion_pred, (tuple, list)) else (motion_pred, 0.0)
                    if m_idx != 0 and m_idx < len(self.point_history_labels): 
                        motion_word = self.point_history_labels[m_idx]
                        word = motion_word

                hand_data.append({"lp": lp, "word": word, "conf": confidence})

            current_frame_word = hand_data[0]["word"]
            if current_frame_word == self.current_proposed_word:
                elapsed = time.time() - self.proposed_word_start_time
                hold_progress = min(1.0, elapsed / self.stabilization_threshold)
                if elapsed >= self.stabilization_threshold:
                    if current_frame_word != self.last_detected_word and self.mode == 0:
                        self.word_buffer.append(current_frame_word)
                        self.last_detected_word = current_frame_word
                        self.update_history_ui(current_frame_word)
            else:
                self.current_proposed_word = current_frame_word
                self.proposed_word_start_time = time.time()
                hold_progress = 0.0

            for data in hand_data:
                self.draw_holographic_ar(debug_image, data["lp"])
                hp = hold_progress if data["word"] == self.current_proposed_word else 0.0
                self.draw_confidence_slider(debug_image, data["lp"], data["conf"], data["word"], hp)
                
            if self.motion_pattern_enabled or self.mode == 2:
                self.draw_point_history(debug_image, self.point_history)

        else:
            self.current_proposed_word = None
            self.proposed_word_start_time = time.time()
            self.capture_next = False 
            if self.motion_pattern_enabled or self.mode == 2:
                self.point_history.append([0, 0]) 

        h, w, ch = debug_image.shape
        qt_img = QImage(debug_image.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
        self.video_display.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # --- UI ACTIONS / HELPERS ---
    def speak_text(self):
        text_to_read = self.nlp_text.text()
        if text_to_read and text_to_read != "Waiting for input..." and "ERROR" not in text_to_read:
            self.tts.say(text_to_read)

    def clear_history(self):
        self.word_buffer.clear()
        self.last_detected_word = None
        self.current_proposed_word = None
        self.point_history.clear()
        self.proposed_word_start_time = time.time()
        self.nlp_text.setText("Waiting for input...")
        for card in self.history_cards:
            card.text_widget.setText("---")

    def on_slider_change(self, value):
        self.stabilization_threshold = value / 10.0
        self.sl_val_label.setText(f"{self.stabilization_threshold:.1f}s")

    def create_ios_card(self, is_latest=False):
        card = QFrame()
        card_shadow = QGraphicsDropShadowEffect(blurRadius=20, xOffset=0, yOffset=8, color=QColor(0,0,0,10))
        card.setGraphicsEffect(card_shadow)
        
        if is_latest:
            card.setFixedHeight(110)
            card.setStyleSheet("background-color: #E5F1FF; border: 2px solid #007AFF; border-radius: 25px;")
            tag_text, tag_color, txt_color, txt_size = "CURRENT SIGN", "#007AFF", "#007AFF", "28px"
        else:
            card.setFixedHeight(90)
            card.setStyleSheet("background-color: white; border: none; border-radius: 25px;")
            tag_text, tag_color, txt_color, txt_size = "PREVIOUS SIGN", "#8E8E93", "#1C1C1E", "22px"

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

    # --- DRAWING / VISUALS ---
    def draw_holographic_ar(self, img, lp):
        paths = [(0,1,2,3,4), (0,5,6,7,8), (0,9,10,11,12), (0,13,14,15,16), (0,17,18,19,20)]
        for path in paths:
            for i in range(len(path)-1):
                cv.line(img, tuple(lp[path[i]]), tuple(lp[path[i+1]]), (255, 122, 0), 2, cv.LINE_AA)
        for pt in lp:
            cv.circle(img, tuple(pt), 5, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(img, tuple(pt), 7, (255, 122, 0), 1, cv.LINE_AA)

    def draw_point_history(self, img, point_history):
        for i, point in enumerate(point_history):
            if point[0] != 0 and point[1] != 0:
                cv.circle(img, tuple(point), 1 + int(i / 2), (152, 251, 152), 2)

    def draw_confidence_slider(self, img, lp, confidence, word, hold_progress=0.0):
        x_coords, y_coords = [pt[0] for pt in lp], [pt[1] for pt in lp]
        min_x, max_x, max_y = min(x_coords), max(x_coords), max(y_coords)
        
        bar_width = max(140, max_x - min_x) 
        start_x = min_x + (max_x - min_x) // 2 - bar_width // 2
        start_y = max_y + 40 
        
        color = (60, 60, 255) if confidence < 0.5 else (0, 165, 255) if confidence < 0.8 else (50, 205, 50)                      
            
        overlay = img.copy()
        cv.rectangle(overlay, (start_x - 15, start_y - 30), (start_x + bar_width + 15, start_y + 22), (28, 28, 30), -1)
        cv.addWeighted(overlay, 0.65, img, 0.35, 0, img)
        
        cv.line(img, (start_x, start_y), (start_x + bar_width, start_y), (80, 80, 80), 6, cv.LINE_AA)
        fill_width = int(bar_width * confidence)
        if fill_width > 0:
            cv.line(img, (start_x, start_y), (start_x + fill_width, start_y), color, 6, cv.LINE_AA)
            
        timer_y = start_y + 10
        cv.line(img, (start_x, timer_y), (start_x + bar_width, timer_y), (80, 80, 80), 3, cv.LINE_AA)
        timer_width = int(bar_width * hold_progress)
        if timer_width > 0:
            timer_color = (0, 200, 255) if hold_progress < 1.0 else (50, 205, 50)
            cv.line(img, (start_x, timer_y), (start_x + timer_width, timer_y), timer_color, 3, cv.LINE_AA)
            
        text = f"{word} ({int(confidence * 100)}%)"
        font, font_scale = cv.FONT_HERSHEY_DUPLEX, 0.45
        text_size = cv.getTextSize(text, font, font_scale, 1)[0]
        cv.putText(img, text, (start_x + (bar_width - text_size[0]) // 2, start_y - 12), font, font_scale, (255, 255, 255), 1, cv.LINE_AA)

    def update_history_ui(self, word):
        self.history_cards[2].text_widget.setText(self.history_cards[1].text_widget.text())
        self.history_cards[1].text_widget.setText(self.history_cards[0].text_widget.text())
        self.history_cards[0].text_widget.setText(word)

    # --- MATH & NORMALIZATION ---
    def calc_landmark_list(self, img, landmarks):
        w, h = img.shape[1], img.shape[0]
        return [[int(lm.x * w), int(lm.y * h)] for lm in landmarks.landmark]

    def pre_process_landmark(self, lp):
        temp = copy.deepcopy(lp)
        bx, by = temp[0][0], temp[0][1]
        for i in range(len(temp)):
            temp[i][0] -= bx
            temp[i][1] -= by
        temp = list(itertools.chain.from_iterable(temp))
        max_v = max(map(abs, temp)) if temp else 1
        return [n / max_v for n in temp]

    def pre_process_point_history(self, image, point_history):
        image_width, image_height = image.shape[1], image.shape[0]
        temp_point_history = copy.deepcopy(point_history)
        base_x, base_y = 0, 0
        if temp_point_history:
            base_x, base_y = temp_point_history[0][0], temp_point_history[0][1]

        for index, point in enumerate(temp_point_history):
            temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
        
        temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
        return temp_point_history

    def run_nlp_reconstruction(self):
        self.status_label.setText("● AI THINKING...")
        self.status_label.setStyleSheet("color: #FF9F0A; font-weight: 800; font-size: 13px; letter-spacing: 1px;")
        self.nlp_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.speech_btn.setEnabled(False) 
        self.nlp_worker = NLPWorker(list(self.word_buffer))
        self.nlp_worker.finished.connect(self.on_nlp_finished)
        self.nlp_worker.start()

    def on_nlp_finished(self, text):
        self.nlp_text.setText(text)
        self.status_label.setText("● AI ENGINE READY")
        self.status_label.setStyleSheet("color: #34C759; font-weight: 800; font-size: 13px; letter-spacing: 1px;")
        self.nlp_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.speech_btn.setEnabled(True) 

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setFont(QFont(".AppleSystemUIFont", 10))
    window = SignSpeakApp()
    window.show()
    sys.exit(app.exec())