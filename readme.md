# Suicidal Tendency Detection using Machine Learning

## Project Overview
Suicidal tendency detection is a crucial task in mental health analysis. This project aims to predict suicidal tendencies based on multiple data sources using machine learning techniques, including Convolutional Neural Networks (CNN) and Random Forest. The goal is to analyze text patterns, speech recognition, and facial gestures to detect possible suicidal tendencies and provide timely alerts to family, friends, or caregivers.

## Introduction
Suicidal thoughts and tendencies can often go undetected, leading to devastating consequences. This project integrates various AI-driven techniques to detect signs of distress and suicidal intent through multiple data sources. By analyzing text, voice, and facial expressions, the system predicts tendencies and provides insights for further intervention.

## Objectives
- Detect suicidal tendencies using a combination of NLP, speech recognition, and image processing.
- Train a deep learning model (CNN) to predict suicidal behavior with high accuracy.
- Compare CNN performance with the Random Forest algorithm.
- Provide a user-friendly interface for data input and predictions.

## System Architecture
1. **Data Collection**: Collect data from various sources, including social media text, voice recordings, and facial expressions.
2. **Preprocessing**: Convert non-numeric data into numerical format using NLP techniques.
3. **Feature Extraction**: Extract key features from text, speech, and images.
4. **Model Training**:
   - Train CNN on labeled dataset
   - Train Random Forest for comparison
5. **Prediction**: Predict whether a person has suicidal tendencies or not.
6. **Visualization**: Generate graphs comparing different models' performances.

## Datasets
The project uses multiple datasets, including:
- **Suicidal Attempt & Stressed Dataset**: Contains records of individuals with and without suicidal thoughts.
- **FER-2013 Dataset**: Used for facial expression recognition.
- **DAIC-WOZ Dataset**: Used for voice pattern analysis.

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - TensorFlow/Keras (Deep Learning)
  - OpenCV (Image Processing)
  - Scikit-learn (ML Algorithms)
  - Pandas & NumPy (Data Manipulation)
  - Matplotlib (Data Visualization)
- **Database**: SQLite
- **Frameworks**:
  - NLP Toolkit
  - CNN & Random Forest for Classification
  
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/SuicidalTendencyDetection.git
   cd SuicidalTendencyDetection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```

## Usage
1. **Upload Dataset**: Select the suicidal attempt dataset for analysis.
2. **Preprocess Data**: Remove missing values and normalize data.
3. **Feature Extraction**: Convert text, speech, and image features to numerical values.
4. **Train Model**:
   - Train CNN using the dataset.
   - Train Random Forest for comparison.
5. **Make Predictions**: Upload test data to predict suicidal tendencies.
6. **View Results**: Check accuracy, precision, recall, and F1-score of models.

## Model Training
- **CNN Training**:
  - Uses Convolutional layers for feature extraction.
  - Trained using the Adam optimizer and categorical cross-entropy loss function.
  - Runs for 70 epochs with a batch size of 16.
- **Random Forest Training**:
  - Uses balanced class weights.
  - Performance compared against CNN results.

## Results
- **CNN Accuracy**: ~92%
- **Random Forest Accuracy**: ~89%
- **Comparison Graph**:
  - Displays precision, recall, F1-score, and accuracy for both models.

## Future Enhancements
- Incorporate real-time monitoring through mobile applications.
- Improve dataset by including real-time social media analysis.
- Implement reinforcement learning to improve model efficiency.

## Contributors
- **M. Tejaswini**  
- **M. Ajay**  
- **Md. Abdul Wajid**  
- **Project Guide**: B. Jyosthna (NRI College, Assistant Professor, CSE Department)

## License
This project is licensed under the MIT License - see the LICENSE file for details.


