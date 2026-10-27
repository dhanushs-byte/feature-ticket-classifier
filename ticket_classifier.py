
import csv
import itertools
import random
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# --------------------------------------------------
# 1. Folders and reproducibility
# --------------------------------------------------
random.seed(42)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

for folder in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "support_tickets.csv"

# --------------------------------------------------
# 2. Build a synthetic dataset of 600 tickets
# --------------------------------------------------
category_templates = {
    "billing": [
        "I was charged twice for my monthly subscription",
        "My invoice shows an incorrect payment amount",
        "I need help with a refund for my recent purchase",
        "The payment failed but money was deducted",
        "Please explain the extra fee on my account",
        "My subscription renewal amount looks incorrect",
    ],
    "technical": [
        "The application crashes whenever I open it",
        "I cannot upload a file to the dashboard",
        "The website displays an unexpected error",
        "The application is extremely slow today",
        "The latest update broke a feature",
        "I receive an error when saving my settings",
    ],
    "account": [
        "I cannot log in to my account",
        "Please help me reset my password",
        "My account has been locked unexpectedly",
        "I am unable to verify my email address",
        "My profile details are not updating",
        "Two factor authentication is not working",
    ],
    "shipping": [
        "My package has not arrived on the expected date",
        "The tracking page shows no delivery updates",
        "My order was delivered to the wrong address",
        "The parcel arrived with damaged packaging",
        "I received the wrong item in my shipment",
        "Please provide an update about my delivery",
    ],
    "cancellation": [
        "I want to cancel my subscription",
        "Please stop the renewal of my current plan",
        "I need to cancel my recent order",
        "My cancellation request has not been processed",
        "I want confirmation that my plan is cancelled",
        "Please explain the cancellation policy",
    ],
}

priority_templates = {
    "low": [
        "There is no immediate urgency",
        "Please review this when convenient",
        "This can wait until the next business day",
        "I would appreciate a routine update",
    ],
    "medium": [
        "I need this resolved soon",
        "This is affecting my regular work",
        "Please investigate and respond today",
        "I need assistance within the next few hours",
    ],
    "high": [
        "URGENT: this is blocking my work completely",
        "Critical issue requiring immediate attention",
        "My business operations are stopped",
        "Please escalate this incident immediately",
    ],
}

# Every category-priority combination gets 40 tickets.
rows = []

for category, priority in itertools.product(
    category_templates, priority_templates
):
    for i in range(40):
        issue = category_templates[category][i % 6]
        urgency = priority_templates[priority][i % 4]
        ticket_id = len(rows) + 1

        details = [
            f"Reference case {ticket_id:04d}",
            f"Customer reports issue number {ticket_id:04d}",
            f"Ticket reference {ticket_id:04d}",
            f"Support request identifier {ticket_id:04d}",
        ]

        text = f"{issue}. {urgency}. {details[i % 4]}."

        rows.append({
            "text": text,
            "category": category,
            "priority": priority,
        })

random.shuffle(rows)

with DATA_FILE.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file, fieldnames=["text", "category", "priority"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("Synthetic dataset created:", DATA_FILE)
print("Original rows:", len(rows))

# --------------------------------------------------
# 3. Clean text and remove duplicate tickets
# --------------------------------------------------
clean_rows = []
seen = set()

with DATA_FILE.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        text = " ".join(row["text"].lower().split()).strip()
        category = row["category"].strip().lower()
        priority = row["priority"].strip().lower()

        if not text:
            continue

        # Remove duplicate text records.
        if text in seen:
            continue

        seen.add(text)
        clean_rows.append({
            "text": text,
            "category": category,
            "priority": priority,
        })

with DATA_FILE.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file, fieldnames=["text", "category", "priority"]
    )
    writer.writeheader()
    writer.writerows(clean_rows)

print("\nClean dataset rows:", len(clean_rows))

# --------------------------------------------------
# 4. Dataset summary and class balance
# --------------------------------------------------
summary_file = OUTPUT_DIR / "dataset_summary.txt"

with summary_file.open("w", encoding="utf-8") as file:
    file.write("SUPPORT TICKET DATASET SUMMARY\n")
    file.write("=" * 35 + "\n")
    file.write(f"Total rows: {len(clean_rows)}\n\n")

    for label in ["category", "priority"]:
        counts = {}
        for row in clean_rows:
            value = row[label]
            counts[value] = counts.get(value, 0) + 1

        file.write(f"{label.upper()} COUNTS\n")
        for name, count in sorted(counts.items()):
            file.write(f"{name}: {count}\n")
        file.write("\n")

print(summary_file.read_text(encoding="utf-8"))

# --------------------------------------------------
# 5. Train / validation / test = 70 / 15 / 15
# Stratify using the combined label to preserve
# category-priority combinations across the splits.
# --------------------------------------------------
texts = [row["text"] for row in clean_rows]
categories = [row["category"] for row in clean_rows]
priorities = [row["priority"] for row in clean_rows]

joint_labels = [
    f"{row['category']}__{row['priority']}"
    for row in clean_rows
]

indices = list(range(len(clean_rows)))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=42,
    stratify=joint_labels,
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
    stratify=[joint_labels[i] for i in temp_idx],
)

def select(values, selected_indices):
    return [values[i] for i in selected_indices]

X_train = select(texts, train_idx)
X_val = select(texts, val_idx)
X_test = select(texts, test_idx)

print("Split sizes:", len(X_train), len(X_val), len(X_test))

# Save split CSV files for reproducibility.
for split_name, selected_indices in [
    ("train", train_idx),
    ("validation", val_idx),
    ("test", test_idx),
]:
    with (DATA_DIR / f"{split_name}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=["text", "category", "priority"]
        )
        writer.writeheader()
        for i in selected_indices:
            writer.writerow(clean_rows[i])

# --------------------------------------------------
# 6. Baseline and character n-gram pipelines
# --------------------------------------------------
def make_pipeline(use_char=False):
    if use_char:
        features = FeatureUnion([
            ("word", TfidfVectorizer(
                ngram_range=(1, 2), sublinear_tf=True
            )),
            ("char", TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=1,
                sublinear_tf=True,
            )),
        ])
    else:
        features = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

    return Pipeline([
        ("tfidf", features),
        ("classifier", LogisticRegression(
            max_iter=2000, random_state=42
        )),
    ])

# --------------------------------------------------
# 7. Tune and evaluate both classification targets
# --------------------------------------------------
all_results = {}
error_records = []

for label_name, all_labels in [
    ("category", categories),
    ("priority", priorities),
]:
    y_train = select(all_labels, train_idx)
    y_val = select(all_labels, val_idx)
    y_test = select(all_labels, test_idx)

    print("\n" + "=" * 50)
    print("TARGET:", label_name.upper())

    # Grid search uses training data only.
    search = GridSearchCV(
        estimator=make_pipeline(use_char=False),
        param_grid={
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "classifier__C": [0.1, 1.0, 10.0],
        },
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print("Best parameters:", search.best_params_)
    print("Best training CV macro F1:", round(search.best_score_, 4))

    # Validation metrics
    val_pred = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average="macro")

    print("\nVALIDATION RESULTS")
    print("Accuracy:", round(val_acc, 4))
    print("Macro F1:", round(val_f1, 4))
    print(classification_report(y_val, val_pred, zero_division=0))

    all_results[label_name] = {
        "model": best_model,
        "validation_accuracy": val_acc,
        "validation_macro_f1": val_f1,
    }

    # Confusion matrix for validation set
    labels_order = sorted(set(all_labels))
    cm = confusion_matrix(y_val, val_pred, labels=labels_order)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=labels_order
    )
    display.plot(xticks_rotation=45, cmap="Blues")
    plt.title(f"{label_name.title()} Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"confusion_matrix_{label_name}.png",
        dpi=150,
    )
    plt.close()

    # Save baseline model and best hyperparameters.
    joblib.dump(
        {
            "model": best_model,
            "target": label_name,
            "best_params": search.best_params_,
        },
        MODEL_DIR / f"{label_name}_model.joblib",
    )

    # Save validation mistakes for error analysis.
    probabilities = best_model.predict_proba(X_val)
    classes = list(best_model.classes_)

    for pos, (actual, predicted) in enumerate(zip(y_val, val_pred)):
        if actual != predicted:
            confidence = float(max(probabilities[pos]))
            error_records.append({
                "target": label_name,
                "text": X_val[pos],
                "actual": actual,
                "predicted": predicted,
                "confidence": round(confidence, 4),
                "possible_reason": (
                    "Overlapping wording, ambiguous context, "
                    "or insufficient training examples"
                ),
            })

    # Bonus: compare word n-grams against word + char n-grams.
    char_search = GridSearchCV(
        estimator=make_pipeline(use_char=True),
        param_grid={
            "tfidf__word__ngram_range": [(1, 1), (1, 2)],
            "classifier__C": [0.1, 1.0, 10.0],
        },
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        refit=True,
    )

    char_search.fit(X_train, y_train)
    char_pred = char_search.best_estimator_.predict(X_val)

    print("\nCHARACTER N-GRAM BONUS")
    print("Word + character accuracy:",
          round(accuracy_score(y_val, char_pred), 4))
    print("Word + character macro F1:",
          round(f1_score(y_val, char_pred, average="macro"), 4))

    # Final test evaluation, after model selection.
    test_pred = best_model.predict(X_test)

    print("\nFINAL TEST RESULTS")
    print("Accuracy:", round(accuracy_score(y_test, test_pred), 4))
    print("Macro F1:", round(
        f1_score(y_test, test_pred, average="macro"), 4
    ))

# --------------------------------------------------
# 8. Save the 10 most confusing examples
# --------------------------------------------------
error_records.sort(key=lambda item: item["confidence"], reverse=True)
error_file = OUTPUT_DIR / "confusing_examples.csv"

with error_file.open("w", newline="", encoding="utf-8") as file:
    fields = [
        "target", "text", "actual", "predicted",
        "confidence", "possible_reason",
    ]
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(error_records[:10])

print("\nSaved error examples:", error_file)
print("Saved models in:", MODEL_DIR)
print("Saved charts in:", OUTPUT_DIR)
print("\nTask 1 and Task 2 completed.")
