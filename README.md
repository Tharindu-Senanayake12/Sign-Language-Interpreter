# 🤟 SignSpeak AI – Real-Time Sign Language Interpreter

**SignSpeak AI** is an advanced real-time sign language interpretation application that bridges the communication gap between sign language users and spoken language.

The system combines **computer vision**, **machine learning**, and **natural language processing** to detect hand gestures, classify them into meaningful words, and reconstruct them into grammatically correct spoken sentences.

The application features a **modern iOS-inspired graphical interface** and also includes a **developer studio** that allows capturing and training custom sign language datasets.

---

# ✨ Key Features

### 🖐 Real-Time Hand Tracking & Classification
Uses **Google MediaPipe** for high-speed and lightweight hand landmark detection.

### 🔍 Dual-Mode Gesture Recognition

**Static Gestures**
- Detects stationary signs such as alphabets or basic words
- Uses spatial landmark coordinates of the hand

**Dynamic Gestures**
- Detects motion-based signs
- Uses **index finger point history tracking**

### 🧠 NLP Sentence Reconstruction
Integrates a **Hugging Face T5 Transformer Model** to convert fragmented gesture keywords into grammatically correct sentences.

Example:

```
ME GO STORE
```

becomes

```
I am going to the store.
```

### 🔊 Native Text-to-Speech
The reconstructed sentence can be spoken aloud using built-in speech synthesis.

### 🧑‍💻 Built-in Developer Studio
A dedicated UI mode for capturing gesture datasets and exporting them directly into CSV files for training custom models.

### 🎨 Modern GUI
Built with **PySide6 (Qt)** featuring:

- Smooth animations
- Drop shadows
- Lock-in gesture detection sliders
- Responsive camera feed

---

# 📸 Screenshots

## Main Translation Dashboard

<div align="center">
<img src="screenshots/main_dashboard.png" width="800">
<p><i>Main translation dashboard with real-time gesture tracking and NLP interpretation.</i></p>
</div>

## Developer Studio Mode

<div align="center">
<img src="screenshots/developer_mode.png" width="800">
<p><i>Developer studio interface for capturing and labeling gesture datasets.</i></p>
</div>

---

# 🛠 Technology Stack

| Category | Technology |
|--------|--------|
| Language | Python |
| Computer Vision | OpenCV, MediaPipe |
| Machine Learning | TensorFlow / Keras (TFLite) |
| NLP | Hugging Face Transformers, PyTorch |
| GUI | PySide6 (Qt) |

---

# 📂 Project Structure

```
sign-language-interpreter/
│
├── app.py
│   Main application (UI, Computer Vision loop, NLP engine)
│
├── model/
│   ├── keypoint_classifier/
│   │   ├── keypoint_classifier.tflite
│   │   ├── keypoint.csv
│   │   └── keypoint_classifier_label.csv
│   │
│   └── point_history_classifier/
│       ├── point_history_classifier.tflite
│       ├── point_history.csv
│       └── point_history_classifier_label.csv
│
├── keypoint_classification.ipynb
│   Training notebook for static gesture models
│
├── point_history_classification.ipynb
│   Training notebook for dynamic gesture models
│
├── requirements.txt
│   Python dependencies
│
└── README.md
```

---

# 🚀 Installation & Setup

## 1 Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/SignSpeak-AI.git
cd SignSpeak-AI
```

## 2 Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

---

## 3 Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure **torch** and **sentencepiece** are installed for the NLP engine.

---

## 4 Run the Application

```bash
python app.py
```

**Note**

During the first run the application will download the **Hugging Face T5 model (~850MB)**.  
Please wait until the **AI THINKING** status completes.

---

# 📖 How to Use

## 🗣 Translation Mode (Default)

1. Stand in front of your webcam
2. Perform a sign gesture
3. Hold the sign until the **Lock-in Timer** fills up
4. The detected word will be added to the history buffer
5. Click **Reconstruct Sentence** to generate a grammatically correct sentence
6. Click **Speech** to hear the output

You can adjust detection speed using the **Lock-in Time Slider**.

---

## 🧑‍💻 Developer Studio (Dataset Logging Mode)

To train the AI with new gestures:

Click **ENTER LOGGING MODE**.

### Keyboard Shortcuts

| Key | Action |
|----|----|
| **K** | Static keypoint logging mode |
| **N** | Dynamic motion logging mode |
| **0–99** | Set gesture label ID |
| **Enter** | Confirm label ID |
| **P** | Capture frame and append to CSV |
| **ESC** | Exit application |

---

# 🧠 Training Custom Models

After capturing enough gesture data:

1. Open one of the notebooks:

```
keypoint_classification.ipynb
```

or

```
point_history_classification.ipynb
```

2. Update the variable:

```
NUM_CLASSES
```

to match the highest label ID used.

3. Ensure label names are correctly updated in:

```
model/.../label.csv
```

4. Run all notebook cells.

The notebook will:

- Train the neural network
- Validate model accuracy
- Export an optimized **.tflite model**

Restart `app.py` and your new gestures will be recognized.

---

# 📜 License

Copyright (c) 2026 Tharindu Senanayake

This project includes some of the codes inspired by work from Kazuhito Takahashi.

---

# 🙏 Acknowledgments

- Hand tracking powered by **Google MediaPipe**
- NLP grammar correction powered by **Hugging Face Transformers**
- Machine learning pipeline inspiration from **Kazuhito00**

---

# 👨‍💻 Author

Created by **[Tharindu Senanayake]**  
2026