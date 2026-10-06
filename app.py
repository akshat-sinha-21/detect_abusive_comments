import tkinter as tk
from tkinter import ttk, messagebox
import joblib

# Load models and vectorizer directly using joblib
vectorizer = joblib.load('tfidf_vectorizer.pkl')
nb_model = joblib.load('naive_bayes_model.pkl')
lr_model = joblib.load('logistic_regression_model.pkl')

def predict_comment():
    comment = text_entry.get("1.0", tk.END).strip()
    if not comment:
        messagebox.showwarning("Empty Input", "Please enter a comment to predict.")
        return

    # Transform raw text directly
    X = vectorizer.transform([comment])

    # Naive Bayes prediction & confidence
    nb_proba = nb_model.predict_proba(X)[0]
    nb_is_toxic = nb_proba[1] >= 0.5
    nb_pred = "Toxic" if nb_is_toxic else "Non-Toxic"
    nb_conf = (nb_proba[1] if nb_is_toxic else nb_proba[0]) * 100

    # Logistic Regression prediction & confidence
    lr_proba = lr_model.predict_proba(X)[0]
    lr_is_toxic = lr_proba[1] >= 0.5
    lr_pred = "Toxic" if lr_is_toxic else "Non-Toxic"
    lr_conf = (lr_proba[1] if lr_is_toxic else lr_proba[0]) * 100

    # Update Naive Bayes label
    nb_color = "#c62828" if nb_is_toxic else "#2e7d32"
    nb_result_label.config(
        text=f"{nb_pred} ({nb_conf:.2f}% confident)",
        foreground=nb_color
    )

    # Update Logistic Regression label
    lr_color = "#c62828" if lr_is_toxic else "#2e7d32"
    lr_result_label.config(
        text=f"{lr_pred} ({lr_conf:.2f}% confident)",
        foreground=lr_color
    )

def clear_text():
    text_entry.delete("1.0", tk.END)
    nb_result_label.config(text="Waiting for input...", foreground="#666666")
    lr_result_label.config(text="Waiting for input...", foreground="#666666")

# Build Simple UI
root = tk.Tk()
root.title("Toxic Comment Classifier")
root.geometry("520x430")
root.resizable(False, False)

# Main container
frame = ttk.Frame(root, padding=20)
frame.pack(fill=tk.BOTH, expand=True)

# Title
title_label = ttk.Label(frame, text="Toxic Comment Classifier", font=("Arial", 16, "bold"))
title_label.pack(pady=(0, 10))

# Input prompt
prompt_label = ttk.Label(frame, text="Enter a comment below:", font=("Arial", 10))
prompt_label.pack(anchor="w", pady=(0, 5))

# Text area
text_entry = tk.Text(frame, height=5, font=("Arial", 10), wrap="word")
text_entry.pack(fill=tk.X, pady=(0, 10))

# Button container
btn_frame = ttk.Frame(frame)
btn_frame.pack(pady=(0, 15))

predict_btn = ttk.Button(btn_frame, text="Predict", command=predict_comment)
predict_btn.pack(side=tk.LEFT, padx=5)

clear_btn = ttk.Button(btn_frame, text="Clear", command=clear_text)
clear_btn.pack(side=tk.LEFT, padx=5)

# Results Box
results_frame = ttk.LabelFrame(frame, text=" Prediction Results ", padding=15)
results_frame.pack(fill=tk.X, pady=5)

# Naive Bayes Row
nb_title = ttk.Label(results_frame, text="Naive Bayes:", font=("Arial", 11, "bold"))
nb_title.grid(row=0, column=0, sticky="w", pady=6)
nb_result_label = ttk.Label(results_frame, text="Waiting for input...", font=("Arial", 11), foreground="#666666")
nb_result_label.grid(row=0, column=1, sticky="w", padx=15, pady=6)

# Logistic Regression Row
lr_title = ttk.Label(results_frame, text="Logistic Regression:", font=("Arial", 11, "bold"))
lr_title.grid(row=1, column=0, sticky="w", pady=6)
lr_result_label = ttk.Label(results_frame, text="Waiting for input...", font=("Arial", 11), foreground="#666666")
lr_result_label.grid(row=1, column=1, sticky="w", padx=15, pady=6)

if __name__ == '__main__':
    root.mainloop()
