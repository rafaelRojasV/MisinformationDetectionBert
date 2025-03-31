#!/usr/bin/env python3
"""
scripts/local_inference.py

A small script to:
 - Load your LoRA + TF-IDF pipeline (vectorizers/scalers).
 - Load the saved model weights.
 - Predict locally on custom text examples (we provide a large sample list).
 - (Optional) If you provide labels, we'll show a confusion matrix & metrics.
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# Adjust these paths to your system:
# --------------------------------------------------------------------
MODEL_DIR = r"C:\Users\Rafael\PycharmProjects\MisinformationDetectionBert\model_artifacts\run_20250331_201945_686076\subrun_config_roberta\peft_fusion_model_hf"
ARTIFACTS_PKL = r"C:\Users\Rafael\PycharmProjects\MisinformationDetectionBert\model_artifacts\run_20250331_201945_686076\subrun_config_roberta\peft_fusion_pipeline_artifacts.pkl"

# --------------------------------------------------------------------
# A large set of sample statements to test the model on.
# Each item has 'cleaned_statement', 'metadata', and an optional 'label'.
# label=0 => "fake", label=1 => "real" (or however your data is defined).
# --------------------------------------------------------------------
SAMPLE_DATA = [
    {
        "cleaned_statement": "Local mayor indicted on charges of bribery, but official court documents are nowhere to be found. Rumor has it the evidence is missing altogether.",
        "metadata": [],
        "label": 0
    },
    {
        "cleaned_statement": "Experts at the WHO declared that the new variant has a 95% chance of being more contagious, confirmed in peer-reviewed journals.",
        "metadata": [],
        "label": 1
    },
    {
        "cleaned_statement": "An online blog claimed NASA admitted the Earth is actually flat, with no official press release to back it up.",
        "metadata": [],
        "label": 0
    },
    {
        "cleaned_statement": "State Department officials reported progress in trade talks, releasing full transcripts online for verification.",
        "metadata": [],
        "label": 1
    },
    {
        "cleaned_statement": "A viral tweet alleges a prominent singer faked her entire pregnancy; no hospital records or reputable source confirms this.",
        "metadata": [],
        "label": 0
    },
    {
        "cleaned_statement": "A newly published academic study shows correlation between daily exercise and reduced anxiety in large clinical trials.",
        "metadata": [],
        "label": 1
    },
    {
        "cleaned_statement": "Suspicious whispers suggest the President canceled all environmental regulations in a secret midnight order, no official statements exist.",
        "metadata": [],
        "label": 0
    },
    {
        "cleaned_statement": "During a televised press conference, the CDC presented data showing vaccine efficacy above 90% in all tested demographics.",
        "metadata": [],
        "label": 1
    },
    {
        "cleaned_statement": "Claims on social media say the moon landing was undone with advanced technology, with zero credible scientific references.",
        "metadata": [],
        "label": 0
    },
    {
        "cleaned_statement": "The Finance Ministry released transparent budget documents detailing recent stimulus allocations for the unemployed.",
        "metadata": [],
        "label": 1
    },
    {
        "cleaned_statement": "A conspiratorial forum insists the government replaced all city water supplies with a mind-control chemical, yet city water test reports show no anomalies.",
        "metadata": [],
        "label": 0
    },
    {
        "cleaned_statement": "Major universities collectively confirmed the results of a large-scale climate change study, publishing their methods and data openly.",
        "metadata": [],
        "label": 1
    },
    {
        "cleaned_statement": "A random Facebook post claims doctors found mythical creatures in newly excavated caves, but there's no credible archaeological record.",
        "metadata": [],
        "label": 0
    },
    {
        "cleaned_statement": "Multiple independent journalists verified official poll results that showed a clear margin of victory in the mayoral race.",
        "metadata": [],
        "label": 1
    },
    {
        "cleaned_statement": "An internet rumor states the Supreme Court secretly ruled to invalidate all state laws, without any mention in the official court docket.",
        "metadata": [],
        "label": 0
    },
    {
        "cleaned_statement": "The National Weather Service publicly released data confirming record-low temperatures last week, consistent across multiple sources.",
        "metadata": [],
        "label": 1
    },
    {
        "cleaned_statement": "One article claims the Vice President pledged to ban all cars nationwide by tomorrow morning, with no actual press coverage or evidence.",
        "metadata": [],
        "label": 0
    },
    {
        "cleaned_statement": "NASA and ESA collaborated on a joint statement showing successful deployment of a new Mars rover, with live video streams from mission control.",
        "metadata": [],
        "label": 1
    },
    {
        "cleaned_statement": "A random website claims hidden microchips are installed in schoolbooks, no documents or official announcements validate this idea.",
        "metadata": [],
        "label": 0
    },
    {
        "cleaned_statement": "An FDA press release confirms the newly approved medication had a 98% success rate in trials, with full peer review included.",
        "metadata": [],
        "label": 1
    },
    {
        "cleaned_statement": "Strange rumor: major corporations collectively decided to remove oxygen from sealed food packages to mind-control the population, no official statement made.",
        "metadata": [],
        "label": 0
    },
    {
        "cleaned_statement": "Local investigators held a public briefing confirming no foul play in the senator's passing, with coroner's report available online.",
        "metadata": [],
        "label": 1
    },
    {
        "cleaned_statement": "A blogger claims the UN forcibly disbanded all national governments, but no resolutions or mainstream news coverage exists.",
        "metadata": [],
        "label": 0
    },
    {
        "cleaned_statement": "Federal records show the newly passed education bill includes increased funding for STEM programs, detailed in official legislative text.",
        "metadata": [],
        "label": 1
    },
    {
        "cleaned_statement": "Instagram screenshots suggest a famous athlete never existed, with no birth records or biography cited, obviously lacking credible proof.",
        "metadata": [],
        "label": 0
    },
    {
        "cleaned_statement": "Independent labs confirmed that a new water purification system meets all health standards, published data in their peer-reviewed study.",
        "metadata": [],
        "label": 1
    },
    {
        "cleaned_statement": "Online rumors keep claiming the President is an alien from Mars, no verified evidence or official genealogical records found.",
        "metadata": [],
        "label": 0
    },
    {
        "cleaned_statement": "Major nonprofit organizations collectively released a statement verifying the success of a global vaccination campaign, with their data open for auditing.",
        "metadata": [],
        "label": 1
    },
    {
        "cleaned_statement": "A single tweet says gravity was outlawed by the Supreme Court, providing no documentation or references to the actual judicial system.",
        "metadata": [],
        "label": 0
    },
    {
        "cleaned_statement": "The Department of Agriculture presented official figures indicating a sharp rise in crop yields, backed by third-party verifications.",
        "metadata": [],
        "label": 1
    },
]

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("===== local_inference.py START =====")

    # 1) Load pipeline artifacts
    logging.info(f"Loading pipeline artifacts from {ARTIFACTS_PKL}")
    artifacts = joblib.load(ARTIFACTS_PKL)
    tfidf_vectorizer = artifacts["tfidf_vectorizer"]
    metadata_scaler = artifacts.get("metadata_scaler", None)

    # 2) Build PEFTBertFusionTransformer
    from transformers import AutoTokenizer
    from src.models.bert_fusion import BertWithNumericFusion
    from src.pipeline.peft_fusion_transformer import PEFTBertFusionTransformer
    from peft import PeftModel, PeftConfig
    import torch

    logging.info(f"Loading LoRA HF model from {MODEL_DIR}")
    peft_config = PeftConfig.from_pretrained(MODEL_DIR)
    # If the config's base model name is None, override it with your known base model
    base_model_name = peft_config.base_model_name_or_path or "roberta-base"
    logging.info(f"Base model resolved to: {base_model_name}")

    # We'll guess tfidf_dim=? meta_dim=? (From your final training logs, meta_dim=0)
    dummy_tfidf_dim = tfidf_vectorizer.max_features or 2000
    dummy_meta_dim = 0

    base_fusion_model = BertWithNumericFusion(
        model_name=base_model_name,
        num_labels=2,  # or however many labels you have
        tfidf_dim=dummy_tfidf_dim,
        meta_dim=dummy_meta_dim,
        class_weights=None
    )

    # Merge LoRA weights
    lora_model = PeftModel.from_pretrained(base_fusion_model, MODEL_DIR)

    # 3) Create scikit-like wrapper
    sk_wrapper = PEFTBertFusionTransformer(
        model_name=base_model_name,
        use_tfidf=True,
        num_labels=2,
        tfidf_max_features=dummy_tfidf_dim,
        scale_tfidf=True,
        scale_metadata=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Attach artifacts
    sk_wrapper.tfidf_vectorizer = tfidf_vectorizer
    tfidf_scaler = artifacts.get("tfidf_scaler", None)
    if tfidf_scaler is not None:
        sk_wrapper.tfidf_scaler = tfidf_scaler

    sk_wrapper.metadata_scaler = metadata_scaler
    sk_wrapper.is_fitted_ = True

    # Attach HF model
    sk_wrapper.model = lora_model
    sk_wrapper.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    # 4) Build sample DataFrame
    sample_df = pd.DataFrame(SAMPLE_DATA)
    logging.info("Sample DataFrame:\n%s", sample_df)

    # 5) Predict
    logging.info("Predicting labels and probabilities for sample data...")
    predicted_labels = sk_wrapper.predict(sample_df)
    predicted_probs = sk_wrapper.predict_proba(sample_df)

    # Show results
    for i, row in sample_df.iterrows():
        text = row["cleaned_statement"]
        actual_label = row.get("label", None)
        pred_label = predicted_labels[i]
        prob_1 = predicted_probs[i][1]  # Probability of label=1
        logging.info(f"\nText: {text}\n  Predicted label={pred_label}, Prob(label=1)={prob_1:.3f}, Actual={actual_label}")

    # 6) If we have ground-truth labels, show confusion matrix
    if "label" in sample_df.columns:
        y_true = sample_df["label"].values
        cm = confusion_matrix(y_true, predicted_labels)
        logging.info(f"\nConfusion Matrix:\n{cm}")

        logging.info("Classification Report:")
        logging.info("\n" + classification_report(y_true, predicted_labels, digits=4))

        # Quick confusion matrix plot
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, [0, 1], rotation=45)
        plt.yticks(tick_marks, [0, 1])
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        out_png = "cm_local_inference.png"
        plt.savefig(out_png, dpi=120)
        logging.info(f"Saved confusion matrix to {out_png}")
    else:
        logging.info("No 'label' column found, skipping confusion matrix.")

    logging.info("===== local_inference.py DONE =====")


if __name__ == "__main__":
    main()
