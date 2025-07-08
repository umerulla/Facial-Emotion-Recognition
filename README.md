# Facial Emotion Recognition 🎭

This project is about recognizing human emotions from facial expressions using a Convolutional Neural Network (CNN) model built from scratch using TensorFlow/Keras.

##  Dataset

- The dataset contains grayscale facial images categorized into 7 emotion classes:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral
- It is organized into two main folders:
  - `train/` - for training data
  - `validation/` - for validation data

## 🛠 Tools and Libraries Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV
- Anaconda (for environment setup)
- Jupyter Notebook

## Project Workflow

### 1. Data Loading
- Used `ImageDataGenerator` for reading images from directories with augmentation applied to training data.

### 2. CNN Model Architecture
- Custom CNN with multiple convolutional layers, max pooling, dropout, and dense layers.
- Final output layer uses `softmax` for classifying into 7 emotion categories.

### 3. Model Training
- Model trained on training data with validation on unseen images.
  
### 4. Model Evaluation
- Plotted training and validation accuracy/loss.
- Saved final model using:
  ```python
  best_model.save("facial_emotion_model.keras")
