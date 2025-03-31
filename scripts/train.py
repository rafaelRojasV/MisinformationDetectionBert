#!/usr/bin/env python3
"""
scripts/train_selective_suppression.py

 - Generates a run_id and creates a top-level directory run_<timestamp+uuid>.
 - Trains multiple models (one per config) in subfolders within that run_dir.
 - Each config's logs, model artifacts, etc. are saved in its own sub-run folder.

Usage:
    python train_selective_suppression.py
    (Make sure to run environment_setup.py first if you haven't already.)
"""

import os
import sys
import logging
import warnings
import uuid
from datetime import datetime

# Filter out certain GPU warnings if needed
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
    # If you want to disable certain PyTorch features:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except ImportError:
    print("PyTorch not installed!", file=sys.stderr)

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

# -----------------
# Our config paths
# -----------------
CONFIG_PATHS = [
    r"C:\Users\Rafael\PycharmProjects\MisinformationDetectionBert\config\config_bert-large.yaml",
    r"C:\Users\Rafael\PycharmProjects\MisinformationDetectionBert\config\config_bert_base.yaml",
    r"C:\Users\Rafael\PycharmProjects\MisinformationDetectionBert\config\config_deberta.yaml",
    r"C:\Users\Rafael\PycharmProjects\MisinformationDetectionBert\config\config_distilbert-base.yaml",
    r"C:\Users\Rafael\PycharmProjects\MisinformationDetectionBert\config\config_roberta.yaml",
]

DATA_PATH = r"C:\Users\Rafael\PycharmProjects\MisinformationDetectionBert\data\processed\pure_political_dataset_truncated.csv"

# ----------------------------------------------------------------------
# Single-config training function
# ----------------------------------------------------------------------
def train_for_config(config_path, sub_run_dir, parent_run_id):
    """
    Loads the given config_path, trains the model, and saves artifacts/logs
    in `sub_run_dir`. The `parent_run_id` is just stored in config for reference.
    """
    import logging, os, yaml, pandas as pd, joblib
    from sklearn.model_selection import train_test_split

    # [MLflow] We'll import mlflow here
    import mlflow

    from src.logger_setup import setup_logging
    from src.config_loader import load_config
    from src.pipeline.peft_fusion_transformer import PEFTBertFusionTransformer
    from src.evaluation.evaluation_utils import evaluate_and_visualize_model_multi, extended_evaluation

    cfg_basename = os.path.splitext(os.path.basename(config_path))[0]

    # Sub-run logs/visuals directories
    os.makedirs(os.path.join(sub_run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(sub_run_dir, "visualizations"), exist_ok=True)

    # Setup sub-run logging
    log_file = os.path.join(sub_run_dir, "logs", f"training_{cfg_basename}.log")
    setup_logging(log_file=log_file, log_level="INFO")  # or "WARN"

    logging.info(f"===== Training with config: {config_path} =====")
    logging.info(f"Results will be saved in: {sub_run_dir}")

    # 1) Load config
    config = load_config(config_path)
    if "system" not in config:
        config["system"] = {}
    config["system"]["run_id"] = parent_run_id  # store for reference

    # 2) Save the loaded config
    config_save_path = os.path.join(sub_run_dir, "visualizations", "config_used.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f)
    logging.info(f"Saved config to {config_save_path}")

    # 3) Load data
    df = pd.read_csv(DATA_PATH)
    logging.info(f"Loaded dataset from {DATA_PATH}, shape={df.shape}")

    if "combined_text" in df.columns:
        df.rename(columns={"combined_text": "cleaned_statement"}, inplace=True)
    if "label" not in df.columns:
        raise ValueError("The dataset must have a 'label' column.")

    df["cleaned_statement"] = df["cleaned_statement"].fillna("")

    # If domain_prob column exists, store each as a single float in metadata
    if "domain_prob" in df.columns:
        df["metadata"] = df["domain_prob"].apply(lambda v: [v])
    else:
        df["metadata"] = [[] for _ in range(len(df))]

    df["row_hash"] = df["cleaned_statement"].apply(hash)

    # Optional sampling
    sample_size = config.get("training", {}).get("sample_size", None)
    if sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logging.info(f"Sampled {sample_size} rows => new shape={df.shape}")

    # 4) Train/val split
    df_train, df_val = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    overlap = set(df_train["row_hash"]).intersection(df_val["row_hash"])
    if overlap:
        logging.warning(f"Possible data leakage! Overlap of {len(overlap)} row_hashes.")
    else:
        logging.info("No overlap between train & val sets.")

    X_train = df_train.drop(columns=["label", "row_hash"])
    y_train = df_train["label"]
    X_val = df_val.drop(columns=["label", "row_hash"])
    y_val = df_val["label"]

    logging.info(f"Training size={len(X_train)}, Val size={len(X_val)}")

    # 5) Extract hyperparameters
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    tfidf_cfg = config.get("tfidf", {})
    meta_cfg = config.get("metadata", {})
    eval_cfg = config.get("evaluation", {})

    # Some hyperparams
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
    import torch
    device = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    class_weights = train_cfg.get("class_weights", None)
    learning_rate = train_cfg.get("learning_rate", 1e-4)
    num_epochs = train_cfg.get("num_train_epochs", 3)
    batch_size = train_cfg.get("batch_size", 8)
    max_length = train_cfg.get("max_length", 128)
    gradient_accum = train_cfg.get("gradient_accumulation_steps", 1)
    label_smoothing = train_cfg.get("label_smoothing_factor", 0.0)
    early_stopping = train_cfg.get("early_stopping_patience", 2)
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

    # [MLflow] We'll group each config training as a single run
    mlflow.set_experiment("SelectiveSuppression_Models")  # or any experiment name you prefer
    with mlflow.start_run(run_name=cfg_basename):
        # [MLflow] Log some config/hyperparams
        mlflow.log_param("config_name", cfg_basename)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("lora_rank", lora_rank)
        mlflow.log_param("lora_alpha", lora_alpha)
        mlflow.log_param("lora_dropout", lora_dropout)
        mlflow.log_param("device", device)
        mlflow.log_param("use_tfidf", use_tfidf)
        mlflow.log_param("sample_size", sample_size if sample_size else "full")

        # 6) Create our model pipeline
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
            learning_rate=float(learning_rate),
            num_train_epochs=num_epochs,
            batch_size=batch_size,
            max_length=max_length,
            output_dir=sub_run_dir,  # Save final model here
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

        # Disable metadata if requested
        if not use_metadata and "metadata" in X_train.columns:
            X_train["metadata"] = [[] for _ in range(len(X_train))]
            X_val["metadata"] = [[] for _ in range(len(X_val))]

        # 7) Fit
        logging.info("Fitting PEFTBertFusionTransformer ...")
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        # 8) Evaluate (validation)
        logging.info("Evaluating on validation set...")
        val_results = evaluate_and_visualize_model_multi(
            model, X_val, y_val,
            model_name=f"{cfg_basename}_val",
            run_dir=sub_run_dir,
            run_id=parent_run_id,
            authenticity_mapping=authenticity_map,
            n_samples_for_pca=n_samples_for_pca,
            thresholds_to_try=thresholds_to_try
        )
        logging.info(f"Val results => {val_results}")

        extended_evaluation(
            model_pipeline=model,
            X=X_val,
            y=y_val,
            run_dir=sub_run_dir,
            model_name=f"{cfg_basename}_val"
        )

        # Optional: Evaluate on training set to detect overfitting
        train_results = evaluate_and_visualize_model_multi(
            model, X_train, y_train,
            model_name=f"{cfg_basename}_train",
            run_dir=sub_run_dir,
            run_id=parent_run_id,
            authenticity_mapping=authenticity_map,
            n_samples_for_pca=min(len(X_train), 500),
            thresholds_to_try=thresholds_to_try
        )
        logging.info(f"Train results => {train_results}")

        train_acc = train_results.get("accuracy", 0.0)
        val_acc = val_results.get("accuracy", 0.0)
        if train_acc - val_acc > 0.20:
            logging.warning(f"Possible overfitting (train_acc={train_acc:.2f} vs val_acc={val_acc:.2f}).")

        # [MLflow] Log final metrics
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("val_accuracy", val_acc)
        # if f1 in results
        if "f1" in val_results:
            mlflow.log_metric("val_f1", val_results["f1"])
        if "f1" in train_results:
            mlflow.log_metric("train_f1", train_results["f1"])

        # 9) Save final model
        huggingface_save_dir = os.path.join(sub_run_dir, "peft_fusion_model_hf")
        model.model.save_pretrained(huggingface_save_dir)
        logging.info(f"LoRA-adapted model saved to {huggingface_save_dir}")

        # Save pipeline artifacts (vectorizers/scalers, etc.)
        pipeline_artifacts_path = os.path.join(sub_run_dir, "peft_fusion_pipeline_artifacts.pkl")
        joblib.dump({
            "tfidf_vectorizer": model.tfidf_vectorizer,
            "metadata_scaler": model.metadata_scaler
        }, pipeline_artifacts_path)
        logging.info(f"Pipeline artifacts saved to {pipeline_artifacts_path}")

        # [MLflow] Optionally log model artifacts if you like
        mlflow.log_artifact(pipeline_artifacts_path, artifact_path="pipeline_artifacts")

        logging.info(f"Done training {cfg_basename}\n")
    # [MLflow] End of with mlflow.start_run => run is automatically ended


# ----------------------------------------------------------------------
# MAIN: 1) generate run_id, 2) create top-level run_dir, 3) loop configs
# ----------------------------------------------------------------------
def main():
    # 1) Generate run_id
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    print(f"🔸 Generated run_id => {run_id}")

    # 2) Create top-level run_dir
    run_dir = os.path.join("model_artifacts", f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Created/using run_dir => {run_dir}")

    # 3) Set up logging for the top-level training script
    #    (We can keep it minimal, since each config also sets up sub-run logs.)
    from src.logger_setup import setup_logging
    top_log_file = os.path.join(run_dir, f"train_selective_suppression_{run_id}.log")
    setup_logging(log_file=top_log_file, log_level="INFO")

    logging.info(f"===== Starting train_selective_suppression.py with run_id={run_id} =====")

    # 4) For each config, create sub-run and train
    for cfg_path in CONFIG_PATHS:
        cfg_name = os.path.splitext(os.path.basename(cfg_path))[0]
        sub_run_dir = os.path.join(run_dir, f"subrun_{cfg_name}")
        os.makedirs(sub_run_dir, exist_ok=True)

        train_for_config(cfg_path, sub_run_dir, run_id)

    logging.info("All configs trained! Done.")


if __name__ == "__main__":
    main()
