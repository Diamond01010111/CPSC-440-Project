# Low-Resource Question Classification on TREC

This project studies low-resource question classification using the TREC question classification dataset. The task is to classify natural-language questions into six coarse-grained categories: `ABBR`, `DESC`, `ENTY`, `HUM`, `LOC`, and `NUM`.

## Project Goal

The goal is to compare how different model families perform when only a limited amount of labelled training data is available. In addition to classification performance, this project also evaluates confidence calibration.

## Models Compared

The notebook compares four models:

- Multinomial Naive Bayes with bag-of-words features
- Logistic Regression with TF-IDF features
- BiLSTM with pretrained GloVe embeddings
- DistilBERT fine-tuned for six-class classification

## Experimental Setup

We create several low-resource training settings from the TREC training set:

- 100 examples
- 500 examples
- 1000 examples
- Full training set

Each setting is split into training and validation data. The official TREC test set is used only for final evaluation.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Macro-F1
- Expected Calibration Error (ECE)
- Reliability diagrams
- Confusion matrix
- Representative misclassified examples

## Runtime

- Platform: Google Colab
- Python: 3.12.13
- GPU: NVIDIA Tesla T4 (when GPU runtime is enabled)

## Files

- `CPSC_440_Final_Project.ipynb`: main notebook (data loading, preprocessing, training, evaluation, plots)
- `requirements.txt`: minimal dependency list
- `requirements_full.txt`: full environment snapshot (`pip freeze`) for exact reproduction

Install minimal dependencies:

```bash
pip install -r requirements.txt
