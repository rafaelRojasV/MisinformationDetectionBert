# src/evaluation/evaluation_utils.py
import os
import logging
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
    Evaluates the model on (X, y), logs metrics, and creates a single figure:
      1) Confusion Matrix
      2) ROC Curve
      3) Precision-Recall Curve
      4) PCA of CLS Embeddings

    Also attempts threshold tuning if:
      - 'thresholds_to_try' is not None
      - model is binary (predict_proba has 2 columns).

    Then:
      - Tests 6 sample phrases (3 "fake" / 3 "real")
      - Displays top-5 TF-IDF words for each phrase in a CSV file

    NOTE: This approach does not fully reflect the LoRA + BERT weighting,
    because your pipeline merges [CLS] embeddings, TF-IDF features, and optional metadata.
    The final MLP sees a fused embedding. Thus, the top TF-IDF tokens only capture
    which words have the highest TF-IDF contribution — not the entire final model logic.
    """
    logging.info(f"===== EVALUATION for: {model_name} =====")

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

    logging.info(f"Accuracy = {acc:.4f}, F1 = {f1:.4f}, Balanced Acc = {bal_acc:.4f}")
    logging.info("Classification Report:\n" + rep)

    # Authenticity mapping (optional usage)
    if (y_proba is not None) and (authenticity_mapping is not None) and (y_proba.shape[1] == 2):
        authenticity_avg = np.mean(y_proba[:, 1] * 100.0)
    else:
        authenticity_avg = None

    # (A) Threshold Tuning (binary only)
    best_thresh = 0.5
    if thresholds_to_try is not None and y_proba is not None and y_proba.shape[1] == 2:
        logging.info("[Threshold Tuning] Trying custom thresholds for binary classification...")
        best_f1 = -1
        y_proba_class1 = y_proba[:, 1]
        for th in thresholds_to_try:
            preds_custom = (y_proba_class1 >= th).astype(int)
            acc_t = accuracy_score(y, preds_custom)
            f1_t = f1_score(y, preds_custom, average='macro')
            bal_acc_t = balanced_accuracy_score(y, preds_custom)
            logging.info(f"  Threshold={th:.2f} => Acc={acc_t:.3f}, F1={f1_t:.3f}, BalAcc={bal_acc_t:.3f}")
            if f1_t > best_f1:
                best_f1 = f1_t
                best_thresh = th
        logging.info(f"Best threshold based on F1: {best_thresh:.2f} (F1={best_f1:.3f})")
    elif thresholds_to_try is not None:
        logging.info("[Threshold Tuning] Skipped (model is multi-class or no predict_proba).")

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

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.set_title("ROC Curve (binary)")
            ax_roc.legend(loc="lower right")

            prec, rec, _ = precision_recall_curve(y, y_proba[:, 1])
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
        # sample data if large
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
            logging.warning(f"[Skipping PCA for {model_name}] transform error: {e}")

        if embeddings is not None and embeddings.ndim == 2 and embeddings.shape[1] >= 2:
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
    logging.info(f"[Saved combined evaluation figure to {fig_path}]")

    # ---------------------------------------------------------------
    # EXTRA: Evaluate 6 sample phrases (3 "fake" / 3 "real") + TF-IDF top words
    # ---------------------------------------------------------------
    sample_phrases = [
        "Fake: The senator was convicted yesterday for crimes that never existed in official records.",
        "Fake: The President secretly abolished Congress, no official statements confirm this action.",
        "Fake: Government stats show negative unemployment which is impossible and unverified.",
        "Real: Independent observers confirmed the election results were correctly tallied this week.",
        "Real: Official budgets indicate increased funding for public schools in the latest quarter.",
        "Real: Health department statements verify the new vaccine has been successfully distributed."
    ]
    # 3 labeled as "fake", 3 labeled as "real" (just for demonstration)
    sample_labels = [0, 0, 0, 1, 1, 1]

    logging.info("===== SAMPLE PHRASE EVALUATION (3 fake, 3 real) =====")
    df_samples = pd.DataFrame({
        "cleaned_statement": sample_phrases,
        "metadata": [[] for _ in range(len(sample_phrases))]  # empty metadata
    })

    # Model predictions
    sample_preds = model_pipeline.predict(df_samples)
    sample_probs = None
    try:
        sample_probs = model_pipeline.predict_proba(df_samples)
    except Exception as ex:
        logging.warning(f"No predict_proba for sample phrases: {ex}")

    for i, phrase in enumerate(sample_phrases):
        pred_label = sample_preds[i]
        real_label = sample_labels[i]
        logging.info(f"Phrase: '{phrase[:80]}...' => Predicted={pred_label}, Actual={real_label}")

    # Retrieve the TF-IDF vocabulary for reference
    tfidf_vectorizer = model_pipeline.tfidf_vectorizer
    vocab_array = np.array(tfidf_vectorizer.get_feature_names_out())

    def get_top_tfidf_words(text, vectorizer, top_k=5):
        """
        Returns the top-k tokens by TF-IDF score for a single text string.
        """
        vec = vectorizer.transform([text])  # shape (1, n_features)
        arr = vec.toarray().flatten()
        top_idx = np.argsort(arr)[::-1][:top_k]
        # Only include tokens with a >0 TF-IDF (otherwise they're not present in text)
        top_words = [(vocab_array[j], arr[j]) for j in top_idx if arr[j] > 0]
        return top_words

    # Build table with top TF-IDF words for each sample
    rows = []
    for i, phrase in enumerate(sample_phrases):
        top_words = get_top_tfidf_words(phrase, tfidf_vectorizer, top_k=5)
        pred_lab = sample_preds[i]
        real_lab = sample_labels[i]
        row = {
            "Sample Index": i,
            "Phrase (truncated)": phrase[:50] + "...",
            "Predicted Label": pred_lab,
            "Actual Label": real_lab,
            "Top TF-IDF Words": "; ".join(f"{w}({s:.2f})" for w, s in top_words)
        }
        rows.append(row)

    df_tfidf_table = pd.DataFrame(rows)
    tfidf_csv_path = os.path.join(run_dir, "visualizations", f"top_tfidf_words_{model_name}_{run_id}.csv")
    df_tfidf_table.to_csv(tfidf_csv_path, index=False)
    logging.info(f"[Saved sample phrases' top TF-IDF words to {tfidf_csv_path}]")

    # Return main evaluation metrics
    return {
        "model": model_name,
        "accuracy": acc,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "authenticity": authenticity_avg,
        "best_threshold": best_thresh
    }


# ------------------------------------------------------------------------
# ADDITIONAL EVALUATIONS: Token-level interpretation + calibration + P/R
# ------------------------------------------------------------------------

def extended_evaluation(
    model_pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    run_dir: str,
    model_name: str
):
    """
    Optional extended evaluations:

    1) Token-level SHAP demonstration (for a small sample) using 6 fixed example phrases.
       - We define a custom SHAP predict function that gracefully handles TF–IDF/metadata or not.
         This way SHAP token-masking won't break the pipeline.
    2) Calibration curve + Brier score (binary only).
    3) Precision@Recall and Recall@Precision at certain operating points.
    4) Multiple SHAP plots (text-based, bar, waterfall, beeswarm, force),
       saving each example's waterfall/force in separate files.

    Results are saved in subfolders of 'visualizations':
      - 'shap_png' for PNG images,
      - 'shap_html' for HTML files (token-level text, force plots).
    """
    import os
    import logging
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    from scipy.special import softmax

    logging.info("===== EXTENDED EVALUATION =====")

    # --- Basic checks on input data ---
    logging.debug(f"[DEBUG] X type: {type(X)}, shape: {X.shape}")
    logging.debug(f"[DEBUG] y type: {type(y)}, length: {len(y)}")
    logging.debug(f"[DEBUG] X columns: {list(X.columns)}")

    if "cleaned_statement" not in X.columns:
        logging.warning("[ExtendedEval] 'cleaned_statement' column not found in X; SHAP might fail.")
    if "metadata" not in X.columns:
        logging.debug("[ExtendedEval] 'metadata' column not found in X. Only relevant if approach uses metadata.")

    # Create main visualization folder if not exists
    vis_dir = os.path.join(run_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    logging.debug(f"[DEBUG] Visualizations directory: {vis_dir}")

    # Also create subfolders for PNG and HTML
    shap_png_dir = os.path.join(vis_dir, "shap_png")
    shap_html_dir = os.path.join(vis_dir, "shap_html")
    os.makedirs(shap_png_dir, exist_ok=True)
    os.makedirs(shap_html_dir, exist_ok=True)

    # -----------------------------------------------------
    # 1) Token-Level SHAP on 6 fixed sample phrases (BATCH)
    # -----------------------------------------------------
    shap_installed = True
    try:
        import shap
        from shap.maskers import Text
    except ImportError:
        shap_installed = False
        logging.warning("[ExtendedEval] SHAP not installed, skipping token-level SHAP demonstration.")

    # Check if model_pipeline has predict_proba
    if not hasattr(model_pipeline, "predict_proba"):
        logging.warning("[ExtendedEval] model_pipeline has no 'predict_proba' method. SHAP might fail.")

    if shap_installed:
        logging.info("[ExtendedEval] Starting SHAP token-level demonstration on fixed sample phrases.")

        # (A) 6 sample phrases (3 fake, 3 real)
        raw_phrases = [
            "Fake: The senator was convicted yesterday for crimes that never existed in official records.",
            "Fake: The President secretly abolished Congress, no official statements confirm this action.",
            "Fake: Government stats show negative unemployment which is impossible and unverified.",
            "Real: Independent observers confirmed the election results were correctly tallied this week.",
            "Real: Official budgets indicate increased funding for public schools in the latest quarter.",
            "Real: Health department statements verify the new vaccine has been successfully distributed."
        ]

        # (B) Convert to normal Python str; skip empty
        sample_phrases = []
        for i, phrase in enumerate(raw_phrases):
            py_str = str(phrase).strip()
            if py_str:
                sample_phrases.append(py_str)
            else:
                logging.warning(f"[SHAP] Skipping empty phrase at index={i}!")

        # (C) Generic SHAP predict function that handles text-only or fusion
        def shap_predict_fusion(text_batch: list) -> np.ndarray:
            """
            text_batch may be strings or lists of tokens (SHAP text masker).
            We'll rejoin tokens if needed, then pass them to the model.
            """
            model = model_pipeline.model
            tokenizer = model_pipeline.tokenizer
            device = model_pipeline._decide_device()

            tfidf_dim = getattr(model_pipeline, "tfidf_dim_", 0)
            meta_dim = getattr(model_pipeline, "meta_dim_", 0)

            final_texts = []
            for item in text_batch:
                if isinstance(item, list):
                    final_texts.append(" ".join(item))
                else:
                    final_texts.append(str(item))

            model.eval()
            model.to(device)

            enc = tokenizer(
                final_texts,
                truncation=True,
                padding=True,
                max_length=model_pipeline.max_length,
                return_tensors="pt"
            ).to(device)

            batch_size = enc["input_ids"].shape[0]

            # Prepare zero-tensors if needed
            tfidf_tensor = None
            if tfidf_dim > 0:
                tfidf_tensor = torch.zeros((batch_size, tfidf_dim), dtype=torch.float32, device=device)
            meta_tensor = None
            if meta_dim > 0:
                meta_tensor = torch.zeros((batch_size, meta_dim), dtype=torch.float32, device=device)

            model_kwargs = {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"]
            }
            if tfidf_tensor is not None:
                model_kwargs["tfidf_feats"] = tfidf_tensor
            if meta_tensor is not None:
                model_kwargs["meta_feats"] = meta_tensor

            with torch.no_grad():
                outputs = model(**model_kwargs)

            logits = outputs.logits.cpu().numpy()
            probs = softmax(logits, axis=1)
            return probs[:, 1]  # Probability of positive (real) class

        # (D) Create SHAP Explainer
        text_masker = Text()  # simpler than Text("split")
        explainer = shap.Explainer(
            shap_predict_fusion,
            masker=text_masker,
            algorithm="auto"
        )

        # (E) Optional length check
        try:
            from transformers import AutoTokenizer
            length_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        except ImportError:
            length_tokenizer = None

        valid_phrases = []
        for i, p in enumerate(sample_phrases):
            if length_tokenizer is not None:
                enc = length_tokenizer(p)
                if len(enc["input_ids"]) > 512:
                    logging.warning(f"[SHAP] Skipping too-long phrase index={i} (>{512} tokens).")
                    continue
            valid_phrases.append(p)

        if not valid_phrases:
            logging.warning("[SHAP] No valid sample phrases => skipping text-level SHAP.")
        else:
            try:
                # shap_values is a shap.Explanation object, shape=(len(valid_phrases),)
                shap_values = explainer(valid_phrases)

                # 1) Token-level text explanation (HTML-based) for each example
                for idx, single_explanation in enumerate(shap_values):
                    text_expl_html = shap.plots.text(single_explanation, display=False)
                    html_out_path = os.path.join(shap_html_dir, f"shap_text_ex_{model_name}_sample{idx}.html")
                    with open(html_out_path, "w", encoding="utf-8") as f:
                        f.write(text_expl_html)
                    logging.info(f"[SHAP] Saved token-level text explanation to {html_out_path}")

                # 2) Create a bar chart & beeswarm for the entire shap_values (global summary)
                bar_png = os.path.join(shap_png_dir, f"shap_bar_{model_name}.png")
                try:
                    fig_bar = shap.plots.bar(shap_values, show=False)
                    plt.savefig(bar_png, dpi=150, bbox_inches="tight")
                    plt.close()
                    logging.info(f"[SHAP] Saved bar plot to {bar_png}")
                except Exception as e:
                    logging.warning(f"[SHAP] Could not create bar plot: {e}")

                beeswarm_png = os.path.join(shap_png_dir, f"shap_beeswarm_{model_name}.png")
                try:
                    fig_bee = shap.plots.beeswarm(shap_values, show=False)
                    plt.savefig(beeswarm_png, dpi=150, bbox_inches="tight")
                    plt.close()
                    logging.info(f"[SHAP] Saved beeswarm plot to {beeswarm_png}")
                except Exception as e:
                    logging.warning(f"[SHAP] Could not create beeswarm plot: {e}")

                # 3) Waterfall & Force Plots for **each** example
                for idx, single_explanation in enumerate(shap_values):
                    # Waterfall: returns a Matplotlib figure
                    try:
                        waterfall_out = os.path.join(shap_png_dir, f"shap_waterfall_{model_name}_sample{idx}.png")
                        fig_wf = shap.plots.waterfall(single_explanation, show=False)
                        plt.savefig(waterfall_out, dpi=150, bbox_inches="tight")
                        plt.close()
                        logging.info(f"[SHAP] Saved waterfall plot for sample {idx}: {waterfall_out}")
                    except Exception as e:
                        logging.warning(f"[SHAP] Could not create waterfall plot for sample {idx}: {e}")

                    # Force: returns a Visualizer => shap.save_html is valid
                    try:
                        force_out = os.path.join(shap_html_dir, f"shap_force_{model_name}_sample{idx}.html")
                        force_fig = shap.plots.force(single_explanation, matplotlib=False)
                        shap.save_html(force_out, force_fig)
                        logging.info(f"[SHAP] Saved force plot (HTML) for sample {idx}: {force_out}")
                    except Exception as e:
                        logging.warning(f"[SHAP] Could not create force plot for sample {idx}: {e}")

            except Exception as e:
                logging.warning(f"[ExtendedEval] SHAP explanation failed: {e}")

    else:
        logging.info("[ExtendedEval] SHAP not installed => skipping token-level SHAP demonstration.")

    # ------------------------------------------------------
    # 2) Calibration Curve + Brier Score (binary only)
    # ------------------------------------------------------
    try:
        if X.shape[0] == 0:
            logging.warning("[ExtendedEval] X is empty. Skipping calibration curve.")
        else:
            y_proba = model_pipeline.predict_proba(X)
            if len(y_proba.shape) != 2:
                logging.warning(f"[ExtendedEval] predict_proba did not return 2D => shape: {y_proba.shape}")
            elif y_proba.shape[1] == 2:
                from sklearn.calibration import calibration_curve

                y_pred_proba = y_proba[:, 1]
                fraction_pos, mean_pred_value = calibration_curve(y, y_pred_proba, n_bins=10)
                brier = np.mean((y - y_pred_proba) ** 2)
                logging.info(f"[ExtendedEval] Brier Score = {brier:.4f}")

                fig_cal, ax_cal = plt.subplots(figsize=(6, 6))
                ax_cal.plot(mean_pred_value, fraction_pos, "o-", label="Model")
                ax_cal.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
                ax_cal.set_xlabel("Predicted Probability")
                ax_cal.set_ylabel("Fraction of positives")
                ax_cal.set_title("Calibration Curve (Binary)")
                ax_cal.legend(loc="best")

                cal_plot_path = os.path.join(shap_png_dir, f"calibration_{model_name}.png")
                fig_cal.savefig(cal_plot_path, dpi=150)
                plt.close(fig_cal)
                logging.info(f"[Saved calibration curve to {cal_plot_path}]")
            else:
                logging.info("[ExtendedEval] Not a binary model => skipping calibration curve.")
    except Exception as e:
        logging.warning(f"[ExtendedEval] Could not compute calibration curve/Brier: {e}")

    # ------------------------------------------------------
    # 3) Precision@Recall and Recall@Precision
    # ------------------------------------------------------
    try:
        if X.shape[0] == 0:
            logging.warning("[ExtendedEval] X is empty. Skipping precision/recall analysis.")
        else:
            y_proba = model_pipeline.predict_proba(X)
            if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
                from sklearn.metrics import precision_score, recall_score

                recall_targets = [0.80, 0.90]
                precision_targets = [0.80, 0.90]

                proba_pos = y_proba[:, 1]
                # Sort descending
                sorted_probs = np.sort(proba_pos)[::-1]

                # For each recall target
                for rt in recall_targets:
                    best_threshold = None
                    for thresh in sorted_probs:
                        preds = (proba_pos >= thresh).astype(int)
                        rec = recall_score(y, preds)
                        if rec >= rt:
                            best_threshold = thresh
                            break
                    if best_threshold is not None:
                        prec_val = precision_score(y, (proba_pos >= best_threshold))
                        logging.info(
                            f"[ExtendedEval] Threshold @ Recall>={rt*100:.0f}% => "
                            f"{best_threshold:.3f}, Precision={prec_val:.3f}"
                        )
                    else:
                        logging.info(f"[ExtendedEval] Could not find threshold for recall>={rt*100:.0f}%")

                # For each precision target
                for pt in precision_targets:
                    best_threshold = None
                    for thresh in sorted_probs:
                        preds = (proba_pos >= thresh).astype(int)
                        prec = precision_score(y, preds)
                        if prec >= pt:
                            best_threshold = thresh
                            break
                    if best_threshold is not None:
                        rec_val = recall_score(y, (proba_pos >= best_threshold))
                        logging.info(
                            f"[ExtendedEval] Threshold @ Precision>={pt*100:.0f}% => "
                            f"{best_threshold:.3f}, Recall={rec_val:.3f}"
                        )
                    else:
                        logging.info(f"[ExtendedEval] Could not find threshold for precision>={pt*100:.0f}%")
            else:
                logging.info("[ExtendedEval] Not a binary model => skipping precision@recall eval.")
    except Exception as e:
        logging.warning(f"[ExtendedEval] Could not compute precision/recall at operating points: {e}")

    logging.info("===== EXTENDED EVALUATION DONE =====")
