#!/usr/bin/env python3
"""
scripts/train_selective_suppression.py

 - (Restored to normal logging) Removes the custom stderr filter that was
   intercepting "FATAL: kernel fmha_cutlassF" lines.
 - Retains the 'sm_120 not compatible' filter if you still want that,
   but you can remove it if desired.
 - Performs data leakage and overfitting checks as well.
"""

import os
import sys
import io
import logging
import warnings
import numpy as np  # Moved up here with the other imports to avoid overshadowing `os`.

# ---------------------------------------------------------
# 0) NO CUSTOM STDERR WRAPPER (RESTORED TO NORMAL)
#    Just use default sys.stderr
# ---------------------------------------------------------

# ---------------------------------------------------------
# 1) SELECTIVE WARNING FILTER (OPTIONAL)
# ---------------------------------------------------------
# This filter will ignore only the "sm_120 ... not compatible" user warning from torch.cuda,
# leaving all other warnings intact.
def sm12_warning_filter(record):
    msg = record.getMessage()
    if "sm_120" in msg and "not compatible with the current PyTorch installation" in msg:
        return False  # ignore that specific warning
    return True

warnings.filterwarnings("default")  # ensure normal warnings are shown
logging.captureWarnings(True)
logger = logging.getLogger("py.warnings")
logger.addFilter(sm12_warning_filter)

# ---------------------------------------------------------
# 2) ENV VARS to reduce extraneous CUDA logs
# ---------------------------------------------------------
os.environ["CUTLASS_LOG_LEVEL"] = "0"
os.environ["CUBLAS_LOG_LEVEL"] = "0"
os.environ["CUDA_DEVICE_DEBUG"] = "0"
os.environ["PYTORCH_CPP_LOG_LEVEL"] = "OFF"
os.environ["ENABLE_NVTX_RANGE"] = "0"
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

# ---------------------------------------------------------
# 3) Now import torch
# ---------------------------------------------------------
try:
    import torch
except ImportError:
    print("PyTorch not installed!", file=sys.stderr)
finally:
    pass
torch.backends.cuda.enable_flash_sdp(False)             # Disable FlashAttention backend
torch.backends.cuda.enable_mem_efficient_sdp(False)     # Disable Memory-Efficient (CUTLASS) backend
torch.backends.cuda.enable_math_sdp(True)
# ---------------------------------------------------------
# 4) Normal script logic starts here
# ---------------------------------------------------------
def main():
    # A) Set up project path
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    env_setup_path = os.path.join(os.path.dirname(__file__), "environment_setup.py")
    if os.path.exists(env_setup_path):
        import scripts.environment_setup as env
    else:
        print("⚠️ environment_setup.py not found, skipping environment checks.")
        env = None

    from src.config_loader import load_config
    from src.logger_setup import setup_logging
    from src.pipeline.peft_fusion_transformer import PEFTBertFusionTransformer
    from src.evaluation.evaluation_utils import evaluate_and_visualize_model_multi
    import pandas as pd
    import yaml
    from sklearn.model_selection import train_test_split
    import joblib  # for model saving

    # C) RUN ID, RUN DIR
    if env:
        run_id = env.run_id
        run_dir = env.run_dir
    else:
        run_id = "manualRunId"
        run_dir = "model_artifacts/run_manual"
        os.makedirs(run_dir, exist_ok=True)

    # D) Load config
    config_path = os.path.join("config", "config.yaml")
    config = load_config(config_path)
    if "system" not in config:
        config["system"] = {}
    config["system"]["run_id"] = run_id

    # E) Logging setup
    run_name = config["system"].get("run_name", "").strip()
    if run_name:
        log_filename = f"training_{run_name}.log"
    else:
        log_filename = f"training_{run_id}.log"

    log_path = os.path.join(run_dir, "logs", log_filename)
    log_level = config["system"].get("logging", {}).get("log_level", "INFO")
    setup_logging(log_file=log_path, log_level=log_level)
    logging.info("🔄 Training script started (restored normal stderr logging).")

    # F) Save config used
    config_save_path = os.path.join(run_dir, "visualizations", "config_used.yaml")
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    with open(config_save_path, "w") as f:
        yaml.dump(config, f)
    logging.info(f"Configuration saved to {config_save_path}")

    # G) Load data
    data_path = os.path.join("data", "processed", "pure_political_dataset_truncated.csv")
    logging.info(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    logging.info(f"Shape of loaded dataset = {df.shape}")
    logging.info(f"Columns: {df.columns.tolist()}")

    if "combined_text" in df.columns:
        df.rename(columns={"combined_text": "cleaned_statement"}, inplace=True)

    if "label" not in df.columns:
        raise ValueError("Dataset must have a 'label' column for classification.")

    df["cleaned_statement"] = df["cleaned_statement"].fillna("")

    # If there's a domain_prob column, make 'metadata' from it
    if "domain_prob" in df.columns:
        meta_vectors = [[val] for val in df["domain_prob"]]
        df["metadata"] = meta_vectors
    else:
        df["metadata"] = [[] for _ in range(len(df))]

    # H) Basic data leakage check
    df["row_hash"] = df.apply(lambda row: hash(row["cleaned_statement"]), axis=1)

    sample_size = config.get("training", {}).get("sample_size", None)
    if sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logging.info(f"Sampled {sample_size} rows => new shape={df.shape}")

    df_train, df_val = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    overlap = set(df_train["row_hash"]).intersection(set(df_val["row_hash"]))
    if overlap:
        logging.warning(f"⚠️ Potential data leakage! Overlapping row_hashes in train & val: {len(overlap)} rows")
    else:
        logging.info("✅ No row overlap detected between train and val sets.")

    X_train = df_train.drop(columns=["label", "row_hash"])
    y_train = df_train["label"]
    X_val   = df_val.drop(columns=["label", "row_hash"])
    y_val   = df_val["label"]
    logging.info(f"Train set size={len(X_train)}, Val set size={len(X_val)}")

    # I) Extract hyperparams from config
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    tfidf_cfg = config.get("tfidf", {})
    meta_cfg  = config.get("metadata", {})
    eval_cfg  = config.get("evaluation", {})

    model_name   = model_cfg.get("name", "bert-base-uncased")
    num_labels   = model_cfg.get("num_labels", 2)
    lora_rank    = model_cfg.get("lora_rank", 8)
    lora_alpha   = model_cfg.get("lora_alpha", 16)
    lora_dropout = model_cfg.get("lora_dropout", 0.1)
    lora_bias    = model_cfg.get("lora_bias", "none")
    lora_targets = model_cfg.get("lora_target_modules", None)

    use_8bit   = train_cfg.get("use_8bit", False)
    gradient_checkpointing = train_cfg.get("gradient_checkpointing", False)
    use_fp16   = train_cfg.get("use_fp16", False)
    device     = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    class_weights = train_cfg.get("class_weights", None)
    learning_rate_str = train_cfg.get("learning_rate", 1e-4)
    num_epochs  = train_cfg.get("num_train_epochs", 3)
    batch_size  = train_cfg.get("batch_size", 8)
    max_length  = train_cfg.get("max_length", 128)
    gradient_accum  = train_cfg.get("gradient_accumulation_steps", 1)
    label_smoothing = train_cfg.get("label_smoothing_factor", 0.0)
    early_stopping  = train_cfg.get("early_stopping_patience", 2)
    output_dir      = train_cfg.get("output_dir", "peft_fusion_output")
    return_logits   = train_cfg.get("return_logits", False)
    infer_batchsize = train_cfg.get("inference_batch_size", 32)

    use_tfidf      = tfidf_cfg.get("use", True)
    tfidf_maxfeat  = tfidf_cfg.get("max_features", 2000)
    tfidf_ngrange  = tuple(tfidf_cfg.get("ngram_range", [1, 3]))
    tfidf_subtf    = tfidf_cfg.get("sublinear_tf", True)
    tfidf_min_df   = tfidf_cfg.get("min_df", 3)
    scale_tfidf    = tfidf_cfg.get("scale", True)

    use_metadata  = meta_cfg.get("use", True)
    scale_meta    = meta_cfg.get("scale", True)

    thresholds_to_try = eval_cfg.get("thresholds_to_try", [0.3, 0.5, 0.7])
    authenticity_map  = eval_cfg.get("authenticity_mapping", None)
    n_samples_for_pca = eval_cfg.get("n_samples_for_pca", 1000)

    # J) Create the pipeline
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

    # K) Drop metadata if not in use
    if not use_metadata and "metadata" in X_train.columns:
        X_train["metadata"] = [[] for _ in range(len(X_train))]
        X_val["metadata"]   = [[] for _ in range(len(X_val))]

    # L) Fit the model
    logging.info("Fitting PEFTBertFusionTransformer on train set...")
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # M) Evaluate on val
    logging.info("🔄 Running evaluation on val set...")
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
    logging.info(f"Val set results: {val_results}")

    # N) Overfitting check: Evaluate on train
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
    logging.info(f"Train set results: {train_results}")

    train_acc = train_results.get("accuracy", 0.0)
    val_acc   = val_results.get("accuracy", 0.0)
    if (train_acc - val_acc) > 0.20:
        logging.warning(f"⚠️ Potential overfitting: train_acc={train_acc:.2f}, val_acc={val_acc:.2f}")

    # O) Save final model
    model_path = os.path.join(run_dir, "peft_fusion_model.pkl")
    joblib.dump(model, model_path)
    logging.info(f"Model saved to: {model_path}")
    logging.info("✅ Training script finished successfully.")

if __name__ == "__main__":
    main()
