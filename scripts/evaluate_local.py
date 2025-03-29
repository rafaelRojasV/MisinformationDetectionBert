#!/usr/bin/env python3
"""
scripts/evaluate.py

Loads a saved model, runs evaluation on new data, and
generates plots/metrics (the 4-in-1 figure).
"""

import os
import logging
import sys

import pandas as pd
import joblib

from src.config_loader import load_config
from src.logger_setup import setup_logging
from src.evaluation.evaluation_utils import evaluate_and_visualize_model_multi

def main():
    # Example: we load from a previous run
    run_id = "some_existing_run_id"
    run_dir = os.path.join("model_artifacts", f"run_{run_id}")

    # 1) Load config & set up logging
    config = load_config("config/config.yaml")
    log_filename = f"evaluation_{run_id}.log"
    log_path = os.path.join(run_dir, "logs", log_filename)
    log_level = config['system'].get('logging', {}).get('log_level', 'INFO')
    setup_logging(log_file=log_path, log_level=log_level)
    logging.info("🔄 Starting evaluation script...")

    # 2) Load model
    model_path = os.path.join(run_dir, "peft_fusion_model.pkl")
    model_pipeline = joblib.load(model_path)
    logging.info(f"Model loaded from {model_path}")

    # 3) Load or prepare test data
    df_test = pd.DataFrame({
        "cleaned_statement": [
            "Completely new test statement",
            "Another unseen example text"
        ],
        "metadata": [[0.5, 0.6], [0.8, 0.9]],
        "label": [0, 1]
    })
    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]

    # 4) Evaluate
    results = evaluate_and_visualize_model_multi(
        model_pipeline,
        X_test,
        y_test,
        model_name="my_peft_model_eval",
        run_dir=run_dir,
        run_id=run_id,
        thresholds_to_try=[0.3, 0.5, 0.7]
    )
    logging.info(f"Evaluation results: {results}")
    logging.info("✅ Finished evaluation script.")

if __name__ == "__main__":
    main()
