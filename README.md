# 🎵 BERT-based Genre Classifier

This project uses a fine-tuned BERT model to classify textual descriptions into one of four music genres. It is implemented using TensorFlow and Hugging Face Transformers.

## 📦 Requirements

- Python 3.8+
- TensorFlow 2.x
- Hugging Face Transformers
- NumPy, Pandas, tqdm

## 🚀 Overview

The pipeline includes:
1. Loading training and test data from JSON files
2. Tokenizing text data using BERT tokenizer (`bert-base-uncased`)
3. Creating a TensorFlow Dataset and batching
4. Fine-tuning a BERT model with a custom classifier head
5. Predicting genre classes for test data
6. Saving the trained model

## 📁 Input Files

- `genre_train.json`: Contains training samples with fields `X` (text) and `Y` (label)
- `genre_test.json`: Contains test samples with field `X` (text only)

## 🧠 Model Architecture

- BERT Base Encoder (`bert-base-uncased`)
- Dense Layer: 64 units, ReLU activation
- Output Layer: 4-class softmax

## 🧪 Training

The training process includes:
- GPU memory growth configuration for stability
- One-hot encoding of labels
- 90/10 train-validation split
- Batch size: 6, Epochs: 5
- Optimizer: Adam (LR = 1e-5, decay = 1e-6)
- Loss: Categorical Crossentropy

## 📤 Inference

- Predicts genre labels for unseen text from `genre_test.json`
- Outputs predictions to `out.csv` with format: `Id,Y`

## 💾 Model Saving

The trained Keras model is saved to the file system (`Bert_transformer_model1`) and can be reloaded for inference.

## 📝 Usage

To run the training and inference:

```bash
python BERTModel.py
```

Ensure that the following files are in the same directory:
- `genre_train.json`
- `genre_test.json`

The model and predictions will be saved as:
- `Bert_transformer_model1/`
- `out.csv`

## 📚 Acknowledgements

- Hugging Face Transformers
- TensorFlow
- BERT (`bert-base-uncased`)

