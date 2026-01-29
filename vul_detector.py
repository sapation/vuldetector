import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification

class VulDetector(torch.nn.Module):
    def __init__(
        self,
        model_name="microsoft/unixcoder-base",
        num_labels=2,
        hidden_dropout=0.1,
        attn_dropout=0.1,
        num_layers=None,
        hidden_size=None,
        num_heads=None,
        freeze_base=False
    ):
        super().__init__()

        # Load config for the chosen checkpoint (UniXcoder, CodeBERT, etc.)
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        config.problem_type = "single_label_classification"
        print(config.max_position_embeddings)
        # Set dropout fields if they exist (different models use different names)
        if hasattr(config, "hidden_dropout_prob"):
            config.hidden_dropout_prob = hidden_dropout
        if hasattr(config, "attention_probs_dropout_prob"):
            config.attention_probs_dropout_prob = attn_dropout
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            config.classifier_dropout = hidden_dropout

        # Optional architecture overrides (only apply if the attribute exists)
        if hidden_size is not None and hasattr(config, "hidden_size"):
            config.hidden_size = hidden_size
        if num_layers is not None and hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = num_layers
        if num_heads is not None and hasattr(config, "num_attention_heads"):
            config.num_attention_heads = num_heads

        # Load the sequence classification head on top of the encoder
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )

        # Freeze base encoder if requested
        if freeze_base:
            base = getattr(self.model, "roberta", None) or getattr(self.model, "encoder", None)
            if base is None:
                # fallback: freeze everything except classifier head
                for name, p in self.model.named_parameters():
                    if "classifier" not in name and "score" not in name:
                        p.requires_grad = False
            else:
                for p in base.parameters():
                    p.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )


class FocalLoss(torch.nn.Module):
    """Multi-class focal loss over logits."""

    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight, dtype=torch.float)
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None
        self.reduction = reduction

    def forward(self, logits, targets):
        if targets.dtype != torch.long:
            targets = targets.long()
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
    