# Emotion Recognition using CNN and OpenCV

This project performs real-time facial emotion detection using a CNN model trained on custom facial image data. It uses OpenCV for webcam video input and face detection, and TensorFlow/Keras for model training and inference.

## 💻 Features
- Face detection using Haar cascades
- Real-time emotion recognition
- Trained CNN with data augmentation
- GUI via OpenCV for live feedback

## 🧠 Emotions Detected
- Angry
- Happy
- Sad
- Surprise

## 📁 Project Structure
```
.
├── emotion_recognition.py   # Inference code using webcam
├── emotion_trained.py       # Model training script
├── emotion_model.h5         # Saved trained model (generated after training)
├── haar_cascade.xml         # Haar cascade classifier for face detection
├── requirements.txt
└── .gitignore
```

## 🧪 How to Train
Update the `train_dir` and `test_dir` paths in `emotion_trained.py` to your local dataset folders. Then run:
```bash
python emotion_trained.py
```

## 🎥 How to Run Emotion Detection
Place the `emotion_model.h5` and `haar_cascade.xml` in the same directory as `emotion_recognition.py`, and then run:
```bash
python emotion_recognition.py
```

## ⚙️ Requirements
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

## 📄 License
This project is for educational and non-commercial use.
