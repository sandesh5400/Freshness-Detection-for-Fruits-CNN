# 🍎 Fruit Freshness Detection using CNN

## 📌 Project Overview
This project is an image classification system that detects whether fruits are fresh or rotten using a Convolutional Neural Network (CNN). The model is trained on a labeled dataset containing different fruit categories.

## 📊 Dataset
Fruits Fresh and Rotten Classification Dataset (Kaggle)

https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

Classes:
- Fresh Apples, Bananas, Oranges  
- Rotten Apples, Bananas, Oranges  

## 🧠 Model Architecture
- Convolutional Layers for feature extraction  
- MaxPooling Layers for downsampling  
- Flatten Layer  
- Dense Fully Connected Layers  
- Dropout Layer to reduce overfitting  

## ⚙️ Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Scikit-learn  

## 🏋️ Training Details
- Image size: 224 x 224  
- Normalization: rescaling (1/255)  
- Optimizer: Adam  
- Loss function: Categorical Crossentropy  
- Epochs: 10  

## 📈 Results
- Test Accuracy: 94.62%  
- Good performance in classifying fresh vs rotten fruits  

## 📷 Output
The model predicts images as:
- Fresh  
- Rotten  

Includes:
- Accuracy graph  
- Loss graph  
- Confusion matrix  
- Sample predictions  

## 🚀 Future Improvements
- Use Transfer Learning (MobileNet, VGG16)  
- Improve accuracy with hyperparameter tuning  
- Deploy as web/mobile application  

## 📚 References
- TensorFlow: https://www.tensorflow.org/  
- Keras: https://keras.io/  
- Scikit-learn: https://scikit-learn.org/  
- Kaggle Dataset: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification  
