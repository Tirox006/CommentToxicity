# Toxic Comment Classification Model

This project is a machine learning model designed to classify toxic comments based on the Jigsaw Toxic Comment Classification Challenge dataset. Using TensorFlow and Gradio, the model predicts the presence of toxic behaviors in text, including categories like toxic, severe toxic, obscene, threat, insult, and identity hate. 

## Table of Contents
- [Installation](#installation)
- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Model Training and Validation](#model-training-and-validation)
- [Evaluation](#evaluation)
- [Deployment with Gradio](#deployment-with-gradio)
- [Conclusion](#conclusion)

## Installation

1. **Clone the Repository**  
   Start by cloning this repository to your local machine and navigating to the project directory.

2. **Install Dependencies**  
   Use `pip` to install all necessary libraries, including TensorFlow, Pandas, Matplotlib, Scikit-learn, and Gradio. 

3. **Prepare the Dataset**  
   Download the dataset from the Jigsaw Toxic Comment Classification Challenge and place it in a folder named `jigsaw-toxic-comment-classification-challenge` within the project directory.

## Project Overview

This project involves building a machine learning model that can detect toxic comments based on text input. It processes the text, trains a neural network model, and allows users to interact with the model using Gradio for easy testing.

## Data Preprocessing

- **Load Data**: The dataset is loaded using Pandas, and the comment texts are extracted as input features.
- **Text Vectorization**: Text data is vectorized using TensorFlow's `TextVectorization` layer, which tokenizes the text and restricts the vocabulary to the top 200,000 words. This is to create a consistent numerical representation of the text that the model can understand.
- **Data Splitting**: The dataset is divided into training, validation, and test sets, with a typical split of 70% for training, 20% for validation, and 10% for testing.

## Model Architecture

- **Embedding Layer**: Converts the input text tokens into dense vectors of a fixed size.
- **Bidirectional LSTM Layer**: Processes the text in both directions to capture contextual information from the surrounding words.
- **Fully Connected Layers**: Several dense layers are used to extract high-level features and learn complex patterns in the data.
- **Output Layer**: The model outputs a sigmoid activation for each label, making it suitable for multi-label classification, as each comment can belong to multiple categories.

## Model Training and Validation

The model is trained on the prepared training dataset and validated on a separate validation set. We use binary cross-entropy as the loss function, which is standard for multi-label classification tasks. 

To monitor model performance, the training and validation loss are plotted using Matplotlib, allowing for easy visualization of learning trends.

## Evaluation

The model is evaluated on the test set using key metrics:
- **Precision**: Measures the accuracy of positive predictions.
- **Recall**: Measures the ability of the model to capture all positive instances.
- **Categorical Accuracy**: Calculates the overall accuracy across all categories.

These metrics help assess the model's performance and reliability in predicting toxic behavior in comments.

## Conclusion

This project demonstrates a neural network-based approach to classifying toxic comments, using a bidirectional LSTM and dense layers to capture and interpret text features. The Gradio interface makes it easy for users to interact with the model, testing its predictions in real time. This project can be further extended with additional preprocessing, data augmentation, or by testing different architectures and optimizers.
