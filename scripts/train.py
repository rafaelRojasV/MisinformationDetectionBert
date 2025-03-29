#!/usr/bin/env python3
"""
scripts/train_selective_suppression.py

 - Restored to normal logging (no custom stderr interception).
 - Saves final model directly into the run_dir from environment_setup.py.
 - Trains with a sample of 200 rows (if config says so), including domain_prob as metadata.
"""

import os
import sys
import logging
import warnings
import numpy as np


def sm12_warning_filter(record):
    msg = record.getMessage()
    if "sm_120" in msg and "not compatible with the current PyTorch installation" in msg:
        return False
    return True


warnings.filterwarnings("default")
logging.captureWarnings(True)
logger = logging.getLogger("py.warnings")
logger.addFilter(sm12_warning_filter)

# Reduce extraneous CUDA logs
os.environ["CUTLASS_LOG_LEVEL"] = "0"
os.environ["CUBLAS_LOG_LEVEL"] = "0"
os.environ["CUDA_DEVICE_DEBUG"] = "0"
os.environ["PYTORCH_CPP_LOG_LEVEL"] = "OFF"
os.environ["ENABLE_NVTX_RANGE"] = "0"
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

try:
    import torch
except ImportError:
    print("PyTorch not installed!", file=sys.stderr)
finally:
    pass

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import importlib.metadata

orig_dist = importlib.metadata.distribution


def patched_distribution(name):
    # For 'sentencepiece' distribution patching
    if name == "sentencepiece":
        try:
            return orig_dist("dbowring-sentencepiece")
        except importlib.metadata.PackageNotFoundError:
            pass
    return orig_dist(name)


importlib.metadata.distribution = patched_distribution


def main():
    # -- 1) Project path and environment setup
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    env_setup_path = os.path.join(os.path.dirname(__file__), "environment_setup.py")
    if os.path.exists(env_setup_path):
        import scripts.environment_setup as env
    else:
        print("[WARN] environment_setup.py not found, skipping environment checks.")
        env = None

    # -- 2) Load config
    from src.config_loader import load_config
    from src.logger_setup import setup_logging
    from src.pipeline.peft_fusion_transformer import PEFTBertFusionTransformer

    # Import both your original and extended evaluation functions:
    from src.evaluation.evaluation_utils import (
        evaluate_and_visualize_model_multi,
        extended_evaluation
    )

    import pandas as pd
    import yaml
    from sklearn.model_selection import train_test_split
    import joblib

    if env:
        run_id = env.run_id
        run_dir = env.run_dir
    else:
        run_id = "manualRunId"
        run_dir = "model_artifacts/run_manual"
        os.makedirs(run_dir, exist_ok=True)

    config_path = os.path.join("config", "config_bert_base.yaml")
    config = load_config(config_path)
    if "system" not in config:
        config["system"] = {}
    config["system"]["run_id"] = run_id

    run_name = config["system"].get("run_name", "").strip()
    if run_name:
        log_filename = f"training_{run_name}.log"
    else:
        log_filename = f"training_{run_id}.log"

    log_path = os.path.join(run_dir, "logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Default to WARN logging to reduce console noise
    log_level = config["system"].get("logging", {}).get("log_level", "WARN")
    setup_logging(log_file=log_path, log_level=log_level)
    logging.warn("[WARN] Training script started (stderr logging restored).")

    # Save the config used
    config_save_path = os.path.join(run_dir, "visualizations", "config_used.yaml")
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    with open(config_save_path, "w") as f:
        yaml.dump(config, f)
    logging.warn(f"[WARN] Configuration saved to {config_save_path}")

    # -- 3) Load data
    data_path = os.path.join("data", "processed", "pure_political_dataset_truncated.csv")
    logging.warn(f"[WARN] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    logging.warn(f"[WARN] Shape of loaded dataset = {df.shape}")
    logging.warn(f"[WARN] Columns: {df.columns.tolist()}")

    if "combined_text" in df.columns:
        df.rename(columns={"combined_text": "cleaned_statement"}, inplace=True)

    if "label" not in df.columns:
        raise ValueError("Dataset must have a 'label' column for classification.")

    df["cleaned_statement"] = df["cleaned_statement"].fillna("")

    # If domain_prob column exists, store each value as metadata
    if "domain_prob" in df.columns:
        meta_vectors = [[val] for val in df["domain_prob"]]
        df["metadata"] = meta_vectors
    else:
        df["metadata"] = [[] for _ in range(len(df))]

    # Create a row-hash to check for data leakage
    df["row_hash"] = df.apply(lambda row: hash(row["cleaned_statement"]), axis=1)

    # Optional sampling (e.g. 200 rows)
    sample_size = config.get("training", {}).get("sample_size", None)
    if sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logging.warn(f"[WARN] Sampled {sample_size} rows => new shape={df.shape}")

    # -- 4) Train/Val split
    from sklearn.model_selection import train_test_split
    df_train, df_val = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    overlap = set(df_train["row_hash"]).intersection(set(df_val["row_hash"]))
    if overlap:
        logging.warning(f"[WARN] Potential data leakage! Overlapping row_hashes: {len(overlap)} rows")
    else:
        logging.warn("[WARN] No row overlap detected between train and val sets.")

    X_train = df_train.drop(columns=["label", "row_hash"])
    y_train = df_train["label"]
    X_val = df_val.drop(columns=["label", "row_hash"])
    y_val = df_val["label"]
    logging.warn(f"[WARN] Train set size={len(X_train)}, Val set size={len(X_val)}")

    # -- 5) Extract config hyperparams
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    tfidf_cfg = config.get("tfidf", {})
    meta_cfg = config.get("metadata", {})
    eval_cfg = config.get("evaluation", {})

    model_name = model_cfg.get("name", "bert-base-uncased")
    num_labels = model_cfg.get("num_labels", 2)
    lora_rank = model_cfg.get("lora_rank", 8)
    lora_alpha = model_cfg.get("lora_alpha", 16)
    lora_dropout = model_cfg.get("lora_dropout", 0.1)
    lora_bias = model_cfg.get("lora_bias", "none")
    lora_targets = model_cfg.get("lora_target_modules", None)

    use_8bit = train_cfg.get("use_8bit", False)
    gradient_checkpointing = train_cfg.get("gradient_checkpointing", False)
    use_fp16 = train_cfg.get("use_fp16", False)
    device = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    class_weights = train_cfg.get("class_weights", None)
    learning_rate_str = train_cfg.get("learning_rate", 1e-4)
    num_epochs = train_cfg.get("num_train_epochs", 3)
    batch_size = train_cfg.get("batch_size", 8)
    max_length = train_cfg.get("max_length", 128)
    gradient_accum = train_cfg.get("gradient_accumulation_steps", 1)
    label_smoothing = train_cfg.get("label_smoothing_factor", 0.0)
    early_stopping = train_cfg.get("early_stopping_patience", 2)
    output_dir = run_dir
    return_logits = train_cfg.get("return_logits", False)
    infer_batchsize = train_cfg.get("inference_batch_size", 32)

    use_tfidf = tfidf_cfg.get("use", True)
    tfidf_maxfeat = tfidf_cfg.get("max_features", 2000)
    tfidf_ngrange = tuple(tfidf_cfg.get("ngram_range", [1, 3]))
    tfidf_subtf = tfidf_cfg.get("sublinear_tf", True)
    tfidf_min_df = tfidf_cfg.get("min_df", 3)
    scale_tfidf = tfidf_cfg.get("scale", True)

    use_metadata = meta_cfg.get("use", True)
    scale_meta = meta_cfg.get("scale", True)

    thresholds_to_try = eval_cfg.get("thresholds_to_try", [0.3, 0.5, 0.7])
    authenticity_map = eval_cfg.get("authenticity_mapping", None)
    n_samples_for_pca = eval_cfg.get("n_samples_for_pca", 1000)

    # -- 6) Create the LoRA pipeline with metadata
    model = PEFTBertFusionTransformer(
        model_name=model_name,
        num_labels=num_labels,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_bias=lora_bias,
        lora_target_modules=lora_targets,
        use_8bit=use_8bit,
        gradient_checkpointing=gradient_checkpointing,
        learning_rate=float(learning_rate_str),
        num_train_epochs=num_epochs,
        batch_size=batch_size,
        max_length=max_length,
        output_dir=output_dir,
        use_fp16=use_fp16,
        return_logits=return_logits,
        use_tfidf=use_tfidf,
        device=device,
        class_weights=class_weights,
        inference_batch_size=infer_batchsize,
        early_stopping_patience=early_stopping,
        tfidf_max_features=tfidf_maxfeat,
        tfidf_ngram_range=tfidf_ngrange,
        tfidf_sublinear_tf=tfidf_subtf,
        tfidf_min_df=tfidf_min_df,
        scale_tfidf=scale_tfidf,
        scale_metadata=scale_meta,
        gradient_accumulation_steps=gradient_accum,
        label_smoothing_factor=label_smoothing
    )

    # If user decided not to use metadata after all, fill it with empty arrays
    if not use_metadata and "metadata" in X_train.columns:
        X_train["metadata"] = [[] for _ in range(len(X_train))]
        X_val["metadata"] = [[] for _ in range(len(X_val))]

    logging.warn("[WARN] Fitting PEFTBertFusionTransformer on train set...")
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # -- 7) Evaluate on validation set
    logging.warn("[WARN] Running evaluation on val set...")
    val_results = evaluate_and_visualize_model_multi(
        model,
        X_val,
        y_val,
        model_name="my_peft_model",
        run_dir=run_dir,
        run_id=run_id,
        authenticity_mapping=authenticity_map,
        n_samples_for_pca=n_samples_for_pca,
        thresholds_to_try=thresholds_to_try
    )
    logging.warn(f"[WARN] Val set results: {val_results}")

    # (Optional) extended evaluation on validation set
    from src.evaluation.evaluation_utils import extended_evaluation
    logging.warn("[WARN] Running extended evaluation on val set...")
    extended_evaluation(
        model_pipeline=model,
        X=X_val,
        y=y_val,
        run_dir=run_dir,
        model_name="my_peft_model"
    )

    # Optionally check for overfitting on train set
    train_results = evaluate_and_visualize_model_multi(
        model,
        X_train,
        y_train,
        model_name="my_peft_model_train_eval",
        run_dir=run_dir,
        run_id=run_id,
        authenticity_mapping=authenticity_map,
        n_samples_for_pca=min(len(X_train), 500),
        thresholds_to_try=thresholds_to_try
    )
    logging.warn(f"[WARN] Train set results: {train_results}")

    train_acc = train_results.get("accuracy", 0.0)
    val_acc = val_results.get("accuracy", 0.0)
    if (train_acc - val_acc) > 0.20:
        logging.warning(f"[WARN] Potential overfitting: train_acc={train_acc:.2f}, val_acc={val_acc:.2f}")

    # -- 8) Save final model
    huggingface_save_dir = os.path.join(run_dir, "peft_fusion_model_hf")
    logging.warn("[WARN] Saving the LoRA-adapted model with save_pretrained()...")
    model.model.save_pretrained(huggingface_save_dir)
    logging.warn(f"[WARN] HF model saved to: {huggingface_save_dir}")

    pipeline_artifacts_path = os.path.join(run_dir, "peft_fusion_pipeline_artifacts.pkl")
    pipeline_artifacts = {
        "tfidf_vectorizer": model.tfidf_vectorizer,
        "metadata_scaler": model.metadata_scaler,
    }
    joblib.dump(pipeline_artifacts, pipeline_artifacts_path)
    logging.warn(f"[WARN] TF-IDF vectorizer & pipeline artifacts saved to: {pipeline_artifacts_path}")

    logging.warn("[WARN] Training script finished successfully.")


if __name__ == "__main__":
    main()
