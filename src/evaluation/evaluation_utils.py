# src/evaluation/evaluation_utils.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, balanced_accuracy_score,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA

sns.set_style("whitegrid")

def batch_predict(model_pipeline, X, batch_size=8):
    """
    Splits X into smaller chunks and calls model_pipeline.predict on each chunk.
    Returns a concatenated array of predictions.
    """
    preds_all = []
    for start_idx in range(0, len(X), batch_size):
        end_idx = start_idx + batch_size
        X_batch = X.iloc[start_idx:end_idx]
        preds_batch = model_pipeline.predict(X_batch)
        preds_all.append(preds_batch)
    return np.concatenate(preds_all, axis=0)

def evaluate_and_visualize_model_multi(
    model_pipeline,
    X,
    y,
    model_name,
    run_dir,
    run_id,
    authenticity_mapping=None,
    n_samples_for_pca=1000,
    thresholds_to_try=None
):
    """
    Evaluates the model on (X, y), prints metrics, and creates a single figure:
      1) Confusion Matrix
      2) ROC Curve
      3) Precision-Recall Curve
      4) PCA of CLS Embeddings

    Also attempts threshold tuning if:
      - 'thresholds_to_try' is not None
      - model is binary (predict_proba has 2 columns).
    """
    print(f"\n===== EVALUATION for: {model_name} =====")

    # 1) Predictions
    y_pred = batch_predict(model_pipeline, X, batch_size=8)

    # 2) Try predict_proba
    try:
        y_proba = model_pipeline.predict_proba(X)
    except Exception:
        y_proba = None

    # 3) Basic metrics
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    bal_acc = balanced_accuracy_score(y, y_pred)
    rep = classification_report(y, y_pred)
    print(f"Accuracy = {acc:.4f}, F1 = {f1:.4f}, Balanced Acc = {bal_acc:.4f}")
    print("Classification Report:\n", rep)

    if (y_proba is not None) and (authenticity_mapping is not None) and (y_proba.shape[1] == 2):
        authenticity_avg = np.mean(y_proba[:, 1] * 100.0)
    else:
        authenticity_avg = None

    # (A) Threshold Tuning (binary only)
    best_thresh = 0.5
    if thresholds_to_try is not None and y_proba is not None and y_proba.shape[1] == 2:
        print("\n[Threshold Tuning] Trying custom thresholds for binary classification...")
        best_f1 = -1
        y_proba_class1 = y_proba[:, 1]
        for th in thresholds_to_try:
            preds_custom = (y_proba_class1 >= th).astype(int)
            acc_t = accuracy_score(y, preds_custom)
            f1_t = f1_score(y, preds_custom, average='macro')
            bal_acc_t = balanced_accuracy_score(y, preds_custom)
            print(f"  Threshold={th:.2f} => Acc={acc_t:.3f}, F1={f1_t:.3f}, BalAcc={bal_acc_t:.3f}")
            if f1_t > best_f1:
                best_f1 = f1_t
                best_thresh = th
        print(f"Best threshold based on F1: {best_thresh:.2f} (F1={best_f1:.3f})\n")
    elif thresholds_to_try is not None:
        print("[Threshold Tuning] Skipped (model is multi-class or no predict_proba).")

    # (B) Figure with Confusion Matrix, ROC, PR, PCA
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Evaluation Visualizations for {model_name}", fontsize=16)

    # 1) Confusion Matrix
    ax_cm = axes[0, 0]
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=np.unique(y), yticklabels=np.unique(y),
        cbar=True, ax=ax_cm
    )
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")

    # 2) ROC & 3) PR
    ax_roc = axes[0, 1]
    ax_pr = axes[1, 0]
    if y_proba is not None:
        n_classes = y_proba.shape[1]
        unique_labels = sorted(np.unique(y))

        from sklearn.metrics import roc_curve, auc, precision_recall_curve
        from sklearn.preprocessing import label_binarize

        if n_classes == 2:
            y_bin = np.array(y)
            fpr, tpr, _ = roc_curve(y_bin, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.set_title("ROC Curve (binary)")
            ax_roc.legend(loc="lower right")

            prec, rec, _ = precision_recall_curve(y_bin, y_proba[:, 1])
            ax_pr.plot(rec, prec, label="Precision-Recall")
            ax_pr.set_title("Precision-Recall (binary)")
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.legend(loc="best")
        else:
            # Multi-class
            y_bin = label_binarize(y, classes=unique_labels)
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                ax_roc.plot(fpr, tpr, label=f"Class {unique_labels[i]}")
            ax_roc.set_title("ROC Curve (multi-class)")
            ax_roc.legend(loc="lower right")

            for i in range(n_classes):
                prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
                ax_pr.plot(rec, prec, label=f"Class {unique_labels[i]}")
            ax_pr.set_title("Precision-Recall (multi-class)")
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.legend(loc="best")
    else:
        ax_roc.text(0.5, 0.5, "No predict_proba() → No ROC", ha='center', va='center')
        ax_pr.text(0.5, 0.5, "No predict_proba() → No PR", ha='center', va='center')

    # 4) PCA of CLS Embeddings
    ax_pca = axes[1, 1]
    if hasattr(model_pipeline, 'transform'):
        if len(X) > n_samples_for_pca:
            X_sample = X.sample(n=n_samples_for_pca, random_state=42)
            y_sample = y.loc[X_sample.index]
        else:
            X_sample = X
            y_sample = y

        try:
            embeddings = model_pipeline.transform(X_sample, return_cls=True)
        except Exception as e:
            embeddings = None
            print(f"[Skipping PCA for {model_name}] transform error: {e}")

        if embeddings is not None and embeddings.ndim == 2 and embeddings.shape[1] >= 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            emb_2d = pca.fit_transform(embeddings)
            y_codes = pd.Series(y_sample).astype('category').cat.codes
            sc = ax_pca.scatter(emb_2d[:, 0], emb_2d[:, 1],
                                c=y_codes, cmap='Spectral', alpha=0.7)
            ax_pca.set_title("PCA of CLS Embeddings")
            ax_pca.set_xlabel("PCA 1")
            ax_pca.set_ylabel("PCA 2")
            fig.colorbar(sc, ax=ax_pca, fraction=0.046, pad=0.04, label="Label (encoded)")
        else:
            ax_pca.text(0.5, 0.5, "No valid embeddings for PCA", ha='center', va='center')
    else:
        ax_pca.text(0.5, 0.5, "No .transform() for PCA", ha='center', va='center')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.join(run_dir, "visualizations"), exist_ok=True)
    fig_path = os.path.join(run_dir, "visualizations", f"all_eval_{model_name}_{run_id}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[Saved combined evaluation figure to {fig_path}]\n")

    return {
        "model": model_name,
        "accuracy": acc,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "authenticity": authenticity_avg,
        "best_threshold": best_thresh
    }
