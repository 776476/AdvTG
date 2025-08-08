import numpy as np
from evaluate import load as load_metric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Metrics for transformer models
def load_transformer_metrics():
    """Load metrics for transformer models from datasets library."""
    accuracy_metric = load_metric("accuracy")
    precision_metric = load_metric("precision")
    recall_metric = load_metric("recall")
    f1_metric = load_metric("f1")
    
    return accuracy_metric, precision_metric, recall_metric, f1_metric

def transformer_metrics(p):
    """Compute metrics for transformer models."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    
    # Load metrics
    accuracy_metric, precision_metric, recall_metric, f1_metric = load_transformer_metrics()

    accuracy = accuracy_metric.compute(
        predictions=predictions, references=labels)
    precision = precision_metric.compute(
        predictions=predictions, references=labels, average="weighted")
    recall = recall_metric.compute(
        predictions=predictions, references=labels, average="weighted")
    f1 = f1_metric.compute(predictions=predictions,
                         references=labels, average="weighted")

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"]
    }

def custom_metrics(preds, labels):
    """Compute metrics for custom models using sklearn."""
    preds = np.argmax(preds, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0)
    
    return accuracy, precision, recall, f1 