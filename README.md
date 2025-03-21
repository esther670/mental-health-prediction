# mental-health-chatbot
## Problem Statement
Mental health issues among university students are a growing concern, often leading to severe consequences if not identified and addressed in time. However, many students struggle to seek help due to stigma or lack of awareness. Traditional methods of mental health assessment require direct intervention, which may not always be feasible.

This project aims to develop an LSTM-based deep learning model for automated mental health text classification. Given a student's written statement about their thoughts or feelings, the model will classify it into one of four mental health categories: Depression, Suicide, Alcoholism, or Drug Abuse.

By leveraging Natural Language Processing (NLP) and Deep Learning, this solution could help universities and mental health professionals detect early signs of mental distress, enabling timely intervention and support.


## Key Objectives:
1. Text Classification: Develop an LSTM-based model to classify students’ statements into relevant mental health categories.
2. Probabilistic Predictions: Instead of strict classification, output a probability score for each category to indicate the likelihood of each mental health issue.
3. Scalability & Real-world Application: Provide a framework that can be extended for use in chatbots, online counseling services, or automated mental health screening tools.


# PROCEDURE
## 1. Preprocessing
- Text Preprocessing – Clean the text by lowercasing, removing punctuation, stopwords, text synonym augmentation and tokenizing.
- Tokenization & Padding – Convert text into sequences and pad them for uniform input size.
- Encode Labels: Since the labels are categorical (Depression, Suicide, Alcoholism, Drugs), they got converted into numerical format using one-hot encoding and label encoding

## 2. Define the Models
- Tested out 3 models i.e LSTM (Long Short Term Memory), GRU (Gated Recurrent Unit), BiLSTM (Bidirectional Long Short Term Memory) and Hybrid model (LSTM + GRU )
- Created a deep learning model using an embedding layer, LSTM layer(s), and dense layers for classification. 
- The general architecture:
Embedding Layer: Converts words into dense vectors.
Spatial dropout: Help prevent overfitting
LSTM/BiLSTM/GRU Layer: Captures sequential patterns in the text.
Normalization layer: to prevent overfitting and improve stability using the ReLU regularization function
Dense Layers: Fully connected layers of 64 neurons that refines the extracted features.
Output layer: uses a softmax activation function to classify the text into one of the multiple categories.

## 3. Compile the Model
- Loss function: categorical_crossentropy (since it's a multi-class classification task).
- Optimizer: Adam (commonly used for NLP tasks).
- Evaluation Metrics: accuracy.

## 4. Train the Models
- Fit the model using the training data.
- Based on the accuracy results on the performance of the 4 models, the hybrid model performed best with an accuracy of 94.64% and loss of 0.71

## 5. Evaluate on Test Set & Generate Predictions
- Fitted the hybrid model to predict label probabilities for the test dataset.


## Data source
https://zindi.africa/competitions/basic-needs-basic-rights-kenya-tech4mentalhealth