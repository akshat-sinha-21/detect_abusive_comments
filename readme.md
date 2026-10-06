# Toxic Comment Classifier

A machine learning project that detects toxic / abusive comments using **TF-IDF** features and two classic classifiers: **Multinomial Naive Bayes** and **Logistic Regression**. It includes a training pipeline and a simple **Tkinter desktop app** for trying out predictions in real time.

## Features

- Text cleaning pipeline: lowercasing, URL/email removal, non-ASCII and special-character stripping, stopword removal, and WordNet lemmatization
- TF-IDF vectorization (top 10,000 features)
- Two models trained and compared side by side:
  - Multinomial Naive Bayes
  - Logistic Regression (`class_weight='balanced'`)
- Evaluation with accuracy, precision, recall, F1, confusion matrix, and classification report
- Desktop GUI showing the prediction and confidence of both models for any comment you type

## Project Structure

```
detect_abusive_comments/
├── app.py                        # Tkinter GUI for live predictions
├── train_ml_models.py            # Cleaning, training, evaluation, model saving
├── train.csv                     # Training data
├── test.csv                      # Test comments
├── test_labels.csv               # Test labels
├── sample_submission.csv         # Sample submission format
├── tfidf_vectorizer.pkl          # Saved TF-IDF vectorizer
├── naive_bayes_model.pkl         # Saved Naive Bayes model
├── logistic_regression_model.pkl # Saved Logistic Regression model
└── output.txt                    # Latest training/evaluation log
```

## Dataset

The data follows the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) format from Kaggle. Each comment has six labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

This project turns it into a **binary** problem: a comment is labeled **toxic (1)** if *any* of the six labels is present, otherwise **non-toxic (0)**. Test rows with a `-1` label (unscored by Kaggle) are dropped.

| Split | Samples used | Non-toxic | Toxic |
|-------|-------------:|----------:|------:|
| Train | 159,465 (after cleaning) | ~89.8% | ~10.2% |
| Test  | 63,440 (after cleaning)  | ~90.2% | ~9.8%  |

## Installation

```bash
git clone https://github.com/akshat-sinha-21/detect_abusive_comments.git
cd detect_abusive_comments
pip install pandas numpy nltk scikit-learn joblib
```

The training script downloads the required NLTK resources (`stopwords`, `wordnet`, `omw-1.4`) automatically. Tkinter ships with most Python installs (on some Linux distros: `sudo apt install python3-tk`).

## Usage

### 1. Train the models (optional, pretrained models are included)

```bash
python train_ml_models.py
```

This cleans the data, trains both models, prints the evaluation report, saves the `.pkl` files, and writes the full log to `output.txt`.

### 2. Run the desktop app

```bash
python app.py
```

Type a comment, click **Predict**, and see each model's verdict (Toxic / Non-Toxic) with a confidence percentage.

## Results

Evaluated on the 63,440 scored test comments:

| Metric    | Naive Bayes | Logistic Regression (balanced) |
|-----------|------------:|-------------------------------:|
| Accuracy  | 0.9350 | 0.8643 |
| Precision | 0.6962 | 0.4154 |
| Recall    | 0.6012 | 0.9320 |
| F1-Score  | 0.6452 | 0.5747 |

**Takeaway:** Naive Bayes is more conservative, giving higher precision and accuracy. Logistic Regression with balanced class weights catches far more toxic comments (recall 93%) but flags more false positives. Which one is better depends on whether missing toxic content or over-flagging is the costlier mistake.

## Tech Stack

- Python
- pandas, NumPy
- NLTK
- scikit-learn
- joblib
- Tkinter

## Future Improvements

- Apply the same text cleaning in `app.py` that is used during training, so inference matches the training distribution
- Tune the decision threshold and try other models (e.g. linear SVM, LightGBM)
- Add multi-label prediction for the six toxicity categories
- Experiment with transformer-based models such as BERT
- Add a web interface or REST API

## Author

**Akshat Sinha**
GitHub: [@akshat-sinha-21](https://github.com/akshat-sinha-21)
