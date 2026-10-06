import pandas as pd
import numpy as np
import re
import unicodedata
import functools
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# Ensure NLTK resources are available
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Precompile regex patterns for faster text cleaning
URL_PATTERN = re.compile(r'http\S+|www\S+|\S+@\S+')
NON_ALPHA_PATTERN = re.compile(r'[^a-z\s]')
WHITESPACE_PATTERN = re.compile(r'\s+')

# Initialize lemmatizer and warm up WordNet
_lemmatizer = WordNetLemmatizer()
_lemmatizer.lemmatize('warmup')

@functools.lru_cache(maxsize=150000)
def _cached_lemmatize(word):
    return _lemmatizer.lemmatize(word)

def clean_text(text, stop_words, lemmatizer=_lemmatizer):
    """
    Cleans text: lowers casing, removes non-ASCII, links, and special characters.
    Lemmatizes and removes stopwords.
    """
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
    text = URL_PATTERN.sub('', text)
    text = NON_ALPHA_PATTERN.sub(' ', text)
    text = WHITESPACE_PATTERN.sub(' ', text).strip()
    words = [_cached_lemmatize(w) for w in text.split() 
             if w not in stop_words and (2 < len(w) < 22)]
    return " ".join(words)

def main():
    output_lines = []
    def log(msg=""):
        print(msg)
        output_lines.append(str(msg))

    log("=" * 60)
    log("  TOXIC COMMENT CLASSIFICATION - MODEL RETRAINING PIPELINE")
    log("=" * 60)

    log("\n1. Loading datasets (train, test, and test_labels)...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    test_labels_df = pd.read_csv('test_labels.csv')

    log(f"   - Raw Train samples : {len(train_df):,}")
    log(f"   - Raw Test comments : {len(test_df):,}")
    log(f"   - Raw Test labels   : {len(test_labels_df):,}")

    log("\n2. Creating Targets and Filtering Test Data...")
    toxic_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Target for training data: toxic = 1 if any toxic label is present, else 0
    train_df['is_toxic'] = train_df[toxic_columns].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    # Merge test set with test labels
    test_merged = pd.merge(test_df, test_labels_df, on='id')
    # Filter out unscored test rows (-1 labels in Kaggle dataset)
    test_valid = test_merged[test_merged['toxic'] != -1].copy().reset_index(drop=True)
    test_valid['is_toxic'] = test_valid[toxic_columns].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

    train_toxic_count = train_df['is_toxic'].sum()
    train_nontoxic_count = len(train_df) - train_toxic_count
    test_toxic_count = test_valid['is_toxic'].sum()
    test_nontoxic_count = len(test_valid) - test_toxic_count

    log(f"   - Valid Test samples (excluding unscored -1 rows): {len(test_valid):,}")
    log(f"   - Train distribution -> Non-Toxic: {train_nontoxic_count:,} ({train_nontoxic_count/len(train_df)*100:.2f}%), "
        f"Toxic: {train_toxic_count:,} ({train_toxic_count/len(train_df)*100:.2f}%)")
    log(f"   - Test distribution  -> Non-Toxic: {test_nontoxic_count:,} ({test_nontoxic_count/len(test_valid)*100:.2f}%), "
        f"Toxic: {test_toxic_count:,} ({test_toxic_count/len(test_valid)*100:.2f}%)")

    log("\n3. Data Cleaning (Removing URLs, special chars, stopwords & lemmatizing)...")
    stop_words = set(stopwords.words('english'))
    
    log("   - Cleaning training comments...")
    train_df['clean_text'] = train_df['comment_text'].apply(lambda x: clean_text(x, stop_words, _lemmatizer))
    
    log("   - Cleaning test comments...")
    test_valid['clean_text'] = test_valid['comment_text'].apply(lambda x: clean_text(x, stop_words, _lemmatizer))

    # Drop rows that became empty after cleaning
    train_df = train_df[train_df['clean_text'].str.len() > 0].reset_index(drop=True)
    test_valid = test_valid[test_valid['clean_text'].str.len() > 0].reset_index(drop=True)

    log(f"   - Cleaned Train samples remaining: {len(train_df):,}")
    log(f"   - Cleaned Test samples remaining : {len(test_valid):,}")

    log("\n4. Feature Extraction: TF-IDF (Text -> Numbers)...")
    X_train_raw = train_df['clean_text']
    y_train = train_df['is_toxic']
    X_test_raw = test_valid['clean_text']
    y_test = test_valid['is_toxic']

    # Fit vectorizer only on training data to prevent data leakage
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    log(f"   - Vocabulary size: {len(vectorizer.get_feature_names_out()):,} features")
    log(f"   - X_train shape  : {X_train.shape}")
    log(f"   - X_test shape   : {X_test.shape}")

    log("\n5. Training Models...")
    log("\n--- Approach 1: Naive Bayes (MultinomialNB) ---")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test)
    log("   MultinomialNB trained successfully.")

    log("\n--- Approach 2: Logistic Regression (class_weight='balanced') ---")
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    log("   LogisticRegression trained successfully.")

    log("\n" + "=" * 60)
    log("  EVALUATION RESULTS ON TEST DATASET")
    log("=" * 60)

    def format_metrics(name, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        cr = classification_report(y_true, y_pred, target_names=['Non-Toxic (0)', 'Toxic (1)'], digits=4)

        log(f"\nModel: {name}")
        log("-" * 45)
        log(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
        log(f"  Precision: {prec:.4f}")
        log(f"  Recall   : {rec:.4f}")
        log(f"  F1-Score : {f1:.4f}")
        log("\n  Confusion Matrix (rows=actual, cols=predicted):")
        log(f"               Pred Non-Toxic    Pred Toxic")
        log(f"  Act Non-Toxic    {cm[0][0]:<15} {cm[0][1]}")
        log(f"  Act Toxic        {cm[1][0]:<15} {cm[1][1]}")
        log("\n  Classification Report:")
        for line in cr.splitlines():
            log(f"  {line}")
        log("-" * 45)
        return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}

    nb_metrics = format_metrics("Naive Bayes (MultinomialNB)", y_test, nb_preds)
    lr_metrics = format_metrics("Logistic Regression (Balanced)", y_test, lr_preds)

    log("\n" + "=" * 60)
    log("  SUMMARY COMPARISON ON TEST DATASET")
    log("=" * 60)
    log(f"  {'Metric':<18} | {'Naive Bayes':<16} | {'Logistic Regression':<20}")
    log(f"  {'-'*18}-+-{'-'*16}-+-{'-'*20}")
    log(f"  {'Accuracy':<18} | {nb_metrics['acc']:<16.4f} | {lr_metrics['acc']:<20.4f}")
    log(f"  {'Precision':<18} | {nb_metrics['prec']:<16.4f} | {lr_metrics['prec']:<20.4f}")
    log(f"  {'Recall':<18} | {nb_metrics['rec']:<16.4f} | {lr_metrics['rec']:<20.4f}")
    log(f"  {'F1-Score':<18} | {nb_metrics['f1']:<16.4f} | {lr_metrics['f1']:<20.4f}")
    log("=" * 60)

    log("\n6. Saving updated models...")
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    joblib.dump(nb_model, 'naive_bayes_model.pkl')
    joblib.dump(lr_model, 'logistic_regression_model.pkl')

    log("   - Saved: tfidf_vectorizer.pkl")
    log("   - Saved: naive_bayes_model.pkl")
    log("   - Saved: logistic_regression_model.pkl")

    # Save output to text file
    output_filename = 'output.txt'
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines) + "\n")
    log(f"\n7. Output successfully saved to: {output_filename}")

if __name__ == '__main__':
    main()

