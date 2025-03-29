import logging
import numpy as np
import pandas as pd
import torch

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from transformers import (
    AutoConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset as HFDataset
from evaluate import load as load_metric

# PEFT imports
from peft import LoraConfig, get_peft_model, TaskType

# Your custom BERT-with-fusion model
from src.models.bert_fusion import BertWithNumericFusion


class PEFTBertFusionTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-like wrapper that:
      - Vectorizes text with BERT (or Roberta), merges in TF-IDF + metadata,
      - Uses LoRA for parameter-efficient fine-tuning,
      - Provides .fit(), .transform(), .predict(), .predict_proba().
    """

    def __init__(
        self,
        # Model / LoRA config
        model_name="bert-base-uncased",
        num_labels=2,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_bias="none",                # whether to train LoRA bias (none, all, lora_only)
        lora_target_modules=None,        # list of module names or None to auto-guess
        use_8bit=False,                  # if True, load model in 8-bit
        gradient_checkpointing=False,    # if True, set training_args.gradient_checkpointing

        # Standard training hyperparams
        learning_rate=1e-4,
        num_train_epochs=3,
        batch_size=8,
        max_length=128,
        output_dir="peft_fusion_output",
        use_fp16=False,
        return_logits=False,
        use_tfidf=True,
        device=None,
        class_weights=None,
        inference_batch_size=32,
        early_stopping_patience=2,

        # TF-IDF params
        tfidf_max_features=2000,
        tfidf_ngram_range=(1, 3),
        tfidf_sublinear_tf=True,
        tfidf_min_df=3,

        # Feature scaling
        scale_tfidf=True,
        scale_metadata=True,

        # Extra fine-tuning parameters
        gradient_accumulation_steps=1,
        label_smoothing_factor=0.0
    ):
        # (A) Model / training config
        self.model_name = model_name
        self.num_labels = num_labels
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_bias = lora_bias
        self.lora_target_modules = lora_target_modules
        self.use_8bit = use_8bit
        self.gradient_checkpointing = gradient_checkpointing

        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.output_dir = output_dir
        self.use_fp16 = use_fp16
        self.return_logits = return_logits
        self.use_tfidf = use_tfidf
        self.device = device
        self.class_weights = class_weights
        self.inference_batch_size = inference_batch_size
        self.early_stopping_patience = early_stopping_patience

        # (B) TF-IDF hyperparams
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.tfidf_sublinear_tf = tfidf_sublinear_tf
        self.tfidf_min_df = tfidf_min_df

        # (C) Scaling booleans
        self.scale_tfidf = scale_tfidf
        self.scale_metadata = scale_metadata

        # (D) Additional hyperparams
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.label_smoothing_factor = label_smoothing_factor

        # (E) Internal placeholders
        self.tokenizer = None
        self.model = None
        self.is_fitted_ = False

        # Renamed to public attributes so your code can reference them directly:
        self.tfidf_vectorizer = None
        self.tfidf_dim_ = 0
        self.meta_dim_ = 0
        self.tfidf_scaler = None
        self.metadata_scaler = None

    def _decide_device(self):
        if self.device:
            d = self.device.lower()
            if d == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            elif d == "mps" and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def _add_tfidf_to_df(self, df, fit_mode=False):
        if not self.use_tfidf:
            return df

        df = df.copy()
        if fit_mode:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.tfidf_max_features,
                ngram_range=self.tfidf_ngram_range,
                sublinear_tf=self.tfidf_sublinear_tf,
                min_df=self.tfidf_min_df
            )
            X_tfidf = self.tfidf_vectorizer.fit_transform(df["cleaned_statement"])
        else:
            if self.tfidf_vectorizer is None:
                raise ValueError("[_add_tfidf_to_df] TF-IDF vectorizer was not fitted yet.")
            X_tfidf = self.tfidf_vectorizer.transform(df["cleaned_statement"])

        df["tfidf"] = list(X_tfidf.toarray())
        return df

    def _prepare_numeric_data(self, df):
        n = len(df)
        if self.use_tfidf and "tfidf" in df.columns:
            tfidf_list = df["tfidf"].values
            tfidf_array = np.stack([np.array(x, dtype=np.float32) for x in tfidf_list])
        else:
            tfidf_array = np.zeros((n, self.tfidf_dim_), dtype=np.float32)

        if "metadata" in df.columns:
            meta_list = df["metadata"].values
            meta_array = np.stack([np.array(x, dtype=np.float32) for x in meta_list])
        else:
            meta_array = np.zeros((n, self.meta_dim_), dtype=np.float32)

        return tfidf_array, meta_array

    def _create_dataset(self, texts, tfidf_array, meta_array, labels=None):
        data_dict = {
            "text": texts,
            "tfidf_feats": tfidf_array,
            "meta_feats": meta_array
        }
        if labels is not None:
            data_dict["labels"] = labels

        ds = HFDataset.from_dict(data_dict)

        def tokenize_fn(example):
            tok = self.tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length
            )
            tok["tfidf_feats"] = example["tfidf_feats"]
            tok["meta_feats"] = example["meta_feats"]
            if labels is not None:
                tok["labels"] = example["labels"]
            return tok

        ds = ds.map(tokenize_fn)
        columns = ["input_ids", "attention_mask", "tfidf_feats", "meta_feats"]
        if labels is not None:
            columns.append("labels")
        ds.set_format(type="torch", columns=columns)
        return ds

    def _collate_fn(self, batch):
        input_ids = torch.stack([x["input_ids"] for x in batch])
        attention_mask = torch.stack([x["attention_mask"] for x in batch])
        tfidf_feats = torch.stack([x["tfidf_feats"] for x in batch])
        meta_feats = torch.stack([x["meta_feats"] for x in batch])

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tfidf_feats": tfidf_feats,
            "meta_feats": meta_feats,
        }
        if "labels" in batch[0]:
            labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
            out["labels"] = labels
        return out

    def fit(self, X_train, y_train=None, X_val=None, y_val=None):
        logging.debug("[fit] Starting LoRA FusionTransformer fit...")

        if y_train is None:
            logging.warning("[PEFTBertFusionTransformer] No labels => skipping training.")
            return self

        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("[PEFTBertFusionTransformer] X_train must be a pd.DataFrame")

        # 1) TF-IDF
        X_train = self._add_tfidf_to_df(X_train, fit_mode=True)

        # 2) Dimensions
        if self.use_tfidf and "tfidf" in X_train.columns:
            example_vec = X_train["tfidf"].iloc[0]
            self.tfidf_dim_ = (
                len(example_vec)
                if isinstance(example_vec, (list, np.ndarray))
                else 0
            )

        if "metadata" in X_train.columns:
            example_meta = X_train["metadata"].iloc[0]
            self.meta_dim_ = (
                len(example_meta)
                if isinstance(example_meta, (list, np.ndarray))
                else 0
            )

        # 3) Convert to NumPy
        tfidf_array, meta_array = self._prepare_numeric_data(X_train)

        # 4) Scale if requested
        if self.scale_tfidf and self.tfidf_dim_ > 0:
            self.tfidf_scaler = StandardScaler()
            tfidf_array = self.tfidf_scaler.fit_transform(tfidf_array)

        if self.scale_metadata and self.meta_dim_ > 0:
            self.metadata_scaler = StandardScaler()
            meta_array = self.metadata_scaler.fit_transform(meta_array)

        # 5) Build base model
        logging.info(
            f"[fit] Building BertWithNumericFusion => tfidf_dim={self.tfidf_dim_}, "
            f"meta_dim={self.meta_dim_}, class_weights={self.class_weights}"
        )

        # If user wants 8-bit loading
        if self.use_8bit:
            logging.info("[fit] Loading base model in 8-bit precision (bitsandbytes).")
            base_model = BertWithNumericFusion.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                tfidf_dim=self.tfidf_dim_,
                meta_dim=self.meta_dim_,
                class_weights=self.class_weights,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            base_model = BertWithNumericFusion(
                model_name=self.model_name,
                num_labels=self.num_labels,
                tfidf_dim=self.tfidf_dim_,
                meta_dim=self.meta_dim_,
                class_weights=self.class_weights
            )

        # 6) LoRA config
        model_config = AutoConfig.from_pretrained(self.model_name)
        model_type = getattr(model_config, "model_type", "").lower()

        if self.lora_target_modules is None:
            # We'll pick a default if user didn't specify
            if model_type == "distilbert":
                target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"]
            elif model_type == "albert":
                target_modules = ["query", "key", "value"]
            elif model_type == "roberta":
                target_modules = ["query", "key", "value", "dense"]
            else:
                target_modules = ["query", "key", "value", "dense"]
        else:
            target_modules = self.lora_target_modules

        lora_cfg = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.lora_bias,
            task_type=TaskType.SEQ_CLS,
            target_modules=target_modules
        )

        self.model = get_peft_model(base_model, lora_cfg)

        # 7) Tokenizer: Force DebertaV2Tokenizer if "deberta-v" in model_name; else use AutoTokenizer
        if "deberta-v" in self.model_name.lower():
            from transformers import DebertaV2Tokenizer
            logging.info("[fit] Using DebertaV2Tokenizer (slow) for DeBERTa models.")
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(
                self.model_name,
                use_fast=False
            )
        else:
            from transformers import AutoTokenizer
            logging.info("[fit] Using AutoTokenizer with use_fast=False.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        # 8) HF train dataset
        text_list = X_train["cleaned_statement"].astype(str).tolist()
        ds_train = self._create_dataset(
            texts=text_list,
            tfidf_array=tfidf_array,
            meta_array=meta_array,
            labels=list(y_train)
        )

        # 9) Validation
        if X_val is not None and y_val is not None:
            X_val = self._add_tfidf_to_df(X_val, fit_mode=False)
            tfidf_val, meta_val = self._prepare_numeric_data(X_val)

            if self.scale_tfidf and self.tfidf_scaler and self.tfidf_dim_ > 0:
                tfidf_val = self.tfidf_scaler.transform(tfidf_val)
            if self.scale_metadata and self.metadata_scaler and self.meta_dim_ > 0:
                meta_val = self.metadata_scaler.transform(meta_val)

            text_val_list = X_val["cleaned_statement"].astype(str).tolist()
            ds_val = self._create_dataset(
                texts=text_val_list,
                tfidf_array=tfidf_val,
                meta_array=meta_val,
                labels=list(y_val)
            )
        else:
            ds_val = None

        # 10) TrainingArguments (use eval_strategy instead of evaluation_strategy)
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch" if ds_val else "no",
            save_strategy="epoch" if ds_val else "no",
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            label_smoothing_factor=self.label_smoothing_factor,
            learning_rate=self.learning_rate,
            logging_steps=50,
            fp16=self.use_fp16,
            load_best_model_at_end=bool(ds_val),
            optim="adamw_torch",
            warmup_ratio=0.06,
            lr_scheduler_type="linear",
            weight_decay=0.01,
            disable_tqdm=False,
            save_total_limit=2
        )

        training_args.gradient_checkpointing = self.gradient_checkpointing

        # 11) Metrics
        acc_metric = load_metric("accuracy")
        f1_metric = load_metric("f1")

        def compute_metrics(eval_pred):
            predictions, labels_ = eval_pred
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            preds = np.argmax(predictions, axis=-1)
            acc = acc_metric.compute(predictions=preds, references=labels_)["accuracy"]
            f1_ = f1_metric.compute(predictions=preds, references=labels_, average="macro")["f1"]
            return {"accuracy": acc, "f1": f1_}

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            data_collator=self._collate_fn,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=self.early_stopping_patience)
            ] if ds_val else None
        )

        device = self._decide_device()
        logging.info(f"[fit] Using device={device}")
        self.model.to(device)

        logging.info("[fit] Starting Trainer.train() ...")
        trainer.train()
        logging.info("[fit] Trainer.train() complete.")

        if ds_val is not None:
            trainer.evaluate()

        self.is_fitted_ = True
        return self

    def transform(self, X, return_cls=True):
        logging.debug(f"[transform] Called with return_cls={return_cls}.")
        if not self.is_fitted_:
            logging.warning("[PEFTBertFusionTransformer] transform() called but model not fitted.")
            return np.zeros((len(X), self.num_labels))

        if not isinstance(X, pd.DataFrame):
            raise ValueError("[PEFTBertFusionTransformer.transform] X must be a pd.DataFrame")

        X = self._add_tfidf_to_df(X, fit_mode=False)
        return self._transform_in_batches(X, return_cls=return_cls, batch_size=self.inference_batch_size)

    def _transform_in_batches(self, X, return_cls=True, batch_size=32):
        self.model.eval()
        device = self._decide_device()
        self.model.to(device)

        n_samples = len(X)
        if n_samples == 0:
            return np.array([])

        # Decide shape
        if return_cls:
            hidden_size = self.model.bert.config.hidden_size
            outputs_arr = np.zeros((n_samples, hidden_size), dtype=np.float32)
        else:
            if self.return_logits:
                outputs_arr = np.zeros((n_samples, self.num_labels), dtype=np.float32)
            else:
                outputs_arr = np.zeros((n_samples,), dtype=np.int32)

        idx = 0
        while idx < n_samples:
            batch_end = min(idx + batch_size, n_samples)
            batch_df = X.iloc[idx:batch_end]

            text_list = batch_df["cleaned_statement"].astype(str).tolist()
            tfidf_array, meta_array = self._prepare_numeric_data(batch_df)

            if self.scale_tfidf and self.tfidf_scaler and self.tfidf_dim_ > 0:
                tfidf_array = self.tfidf_scaler.transform(tfidf_array)
            if self.scale_metadata and self.metadata_scaler and self.meta_dim_ > 0:
                meta_array = self.metadata_scaler.transform(meta_array)

            tokenized = self.tokenizer(
                text_list,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            tfidf_tensor = torch.tensor(tfidf_array, dtype=torch.float32, device=device)
            meta_tensor = torch.tensor(meta_array, dtype=torch.float32, device=device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    tfidf_feats=tfidf_tensor,
                    meta_feats=meta_tensor
                )

            if return_cls:
                cls_emb = outputs.hidden_states
                outputs_arr[idx:batch_end, :] = cls_emb.cpu().numpy()
            else:
                logits = outputs.logits.cpu().numpy()
                if self.return_logits:
                    outputs_arr[idx:batch_end, :] = logits
                else:
                    preds = np.argmax(logits, axis=-1)
                    outputs_arr[idx:batch_end] = preds

            idx = batch_end

        return outputs_arr

    def predict(self, X):
        logging.debug("[predict] Called => class IDs.")
        if not self.is_fitted_:
            logging.warning("[PEFTBertFusionTransformer] predict() but model not fitted.")
            return np.zeros(len(X), dtype=int)

        old_setting = self.return_logits
        self.return_logits = False
        preds = self.transform(X, return_cls=False)
        self.return_logits = old_setting
        return preds

    def predict_proba(self, X):
        logging.debug("[predict_proba] Called => returning probabilities.")
        if not self.is_fitted_:
            logging.warning("[PEFTBertFusionTransformer] not fitted, returning zeros.")
            return np.zeros((len(X), self.num_labels), dtype=float)

        old_setting = self.return_logits
        self.return_logits = True
        logits = self.transform(X, return_cls=False)
        self.return_logits = old_setting

        from scipy.special import softmax
        return softmax(logits, axis=1)
