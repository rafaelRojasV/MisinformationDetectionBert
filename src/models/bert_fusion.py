# src/models/bert_fusion.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput

class BertWithNumericFusion(nn.Module):
    """
    A PyTorch model that:
      - Uses a Transformer encoder (e.g. BERT/Roberta) => [CLS] embedding
      - Concatenates that embedding with TF-IDF + metadata vectors
      - Feeds the fused vector into a small MLP classifier
      - Optionally applies weighted cross-entropy if class_weights is given
    """
    def __init__(
        self,
        model_name,
        num_labels,
        tfidf_dim,
        meta_dim,
        class_weights=None
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels

        self.class_weights = class_weights

        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size  # e.g. 768 or 1024

        # Fused classifier
        fusion_input_dim = hidden_size + tfidf_dim + meta_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        tfidf_feats=None,
        meta_feats=None,
        labels=None,
        **kwargs
    ):
        # Remove extraneous kwargs
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("num_items_in_batch", None)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        concat_list = [cls_embedding]
        if tfidf_feats is not None and tfidf_feats.shape[1] > 0:
            concat_list.append(tfidf_feats.float())
        if meta_feats is not None and meta_feats.shape[1] > 0:
            concat_list.append(meta_feats.float())

        fused = torch.cat(concat_list, dim=1)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            if self.class_weights is not None:
                weight_tensor = torch.tensor(
                    self.class_weights, dtype=torch.float32, device=logits.device
                )
                loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
            else:
                loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=cls_embedding,
            attentions=outputs.attentions
        )
