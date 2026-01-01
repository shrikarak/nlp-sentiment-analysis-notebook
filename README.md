# Sentiment Analysis with LSTMs in TensorFlow/Keras

This repository contains a Jupyter Notebook that demonstrates how to build and train a Recurrent Neural Network (RNN) for sentiment analysis. The project uses a Bidirectional LSTM (Long Short-Term Memory) network to classify movie reviews from the IMDB dataset as either positive or negative.

## Overview

Sentiment analysis is a fundamental task in Natural Language Processing (NLP) with wide-ranging applications, from analyzing customer feedback to monitoring brand perception on social media. This notebook provides a complete, end-to-end example of how to tackle this problem using deep learning.

The model is built with `tf.keras` and processes sequences of text to learn the contextual relationships between words, enabling it to make accurate sentiment predictions.

## Key Concepts Demonstrated

- **Text Preprocessing**: Loading a pre-tokenized dataset and standardizing sequence lengths using padding.
- **Word Embeddings**: Using an `Embedding` layer to represent words as dense vectors, which are learned during training.
- **Recurrent Neural Networks (RNNs)**: Building a model with a `Bidirectional LSTM` layer to effectively process sequential data like text.
- **Model Training and Evaluation**: Compiling and training the model, and then evaluating its performance on an unseen test set.
- **Real-world Prediction**: Creating a utility function to preprocess and predict the sentiment of new, raw text sentences.

## Tech Stack

- **TensorFlow / Keras**: The core deep learning framework used to build and train the model.
- **NumPy**: For numerical operations.
- **Matplotlib**: For plotting and visualizing the training history.
- **Jupyter Notebook**: For interactive development and documentation.

## Setup and Installation

Follow these steps to set up and run the project.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/shrikarak/nlp-sentiment-analysis-notebook.git
    cd nlp-sentiment-analysis-notebook
    ```

2.  **Create and Activate a Virtual Environment** (Recommended)
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## How to Use the Notebook

1.  **Launch Jupyter Notebook**
    ```bash
    jupyter notebook
    ```

2.  **Run the Cells**
    - Open the `sentiment_analysis_imdb.ipynb` file in your browser.
    - Execute the cells sequentially from top to bottom.
    - The IMDB dataset will be downloaded automatically by Keras the first time you run the data loading cell.
    - The notebook will train the model, evaluate it, and finally show you how to test it with your own sentences.

## Copyright and License

*   **Copyright (c) 2026 Shrikara Kaudambady.**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.
