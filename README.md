# Sign Language Detection System

Real-time American Sign Language (ASL) recognition using computer vision and machine learning.

## Overview

This project detects and translates ASL letters (A-Z) into text in real-time using a webcam. It uses MediaPipe for hand tracking and Random Forest for classification, achieving 95%+ accuracy.

## Features

- Real-time hand sign detection
- Recognition of 26 ASL letters
- Text composition with space/backspace/clear commands
- Visual feedback with hand tracking overlay
- 30+ FPS performance

## Tech Stack

- **Python** - Core programming language
- **OpenCV** - Video processing and display
- **MediaPipe** - Hand landmark detection
- **Scikit-learn** - Random Forest classifier
- **NumPy** - Data processing

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/sign-language-detector.git
cd sign-language-detector

# Install dependencies
pip install opencv-python mediapipe scikit-learn numpy
```

## Usage

1. **Collect training data**: `python collect_imgs.py`
2. **Process dataset**: `python create_dataset.py`
3. **Train model**: `python train_classifier.py`
4. **Run detection**: `python inference_classifier.py`

## Project Structure

```
├── collect_imgs.py          # Collect training images
├── create_dataset.py        # Extract hand landmarks
├── train_classifier.py      # Train Random Forest model
├── inference_classifier.py  # Real-time detection
├── data.pickle             # Processed features
└── model.p                 # Trained model
```

## Results

- **Accuracy**: 95%+ on test set
- **Training Data**: 100+ images per sign
- **Processing Speed**: Real-time at 30+ FPS
- **Model**: Random Forest with 200 trees

## Demo

[![Watch the video](https://img.youtube.com/vi/hATuQtdwcQ0/0.jpg)](https://www.youtube.com/watch?v=hATuQtdwcQ0)
## Future Improvements

- Add support for words and phrases
- Implement deep learning models
- Create web interface
- Support for multiple sign languages

---

**Note**: Developed as a computer vision and machine learning project to explore real-time gesture recognition.





