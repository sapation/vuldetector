import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

class VulDetector(nn.Module):
    def __init__(
        self,
        model_name="microsoft/unixcoder-base",
        num_labels=2,
        hidden_dropout=0.1,
        attn_dropout=0.1,
        num_layers=None,
        hidden_size=None,
        num_heads=None,
        freeze_base=False,
        head_type="mlp",          # "linear" or "mlp"
        pooling="cls",            # "cls" or "mean"
    ):
        super().__init__()

        # Load config (encoder config)
        config = AutoConfig.from_pretrained(model_name)
        print("max_position_embeddings:", getattr(config, "max_position_embeddings", None))

        # set dropouts if present
        if hasattr(config, "hidden_dropout_prob"):
            config.hidden_dropout_prob = hidden_dropout
        if hasattr(config, "attention_probs_dropout_prob"):
            config.attention_probs_dropout_prob = attn_dropout

        # Optional architecture overrides (only if attribute exists)
        if hidden_size is not None and hasattr(config, "hidden_size"):
            config.hidden_size = hidden_size
        if num_layers is not None and hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = num_layers
        if num_heads is not None and hasattr(config, "num_attention_heads"):
            config.num_attention_heads = num_heads

        self.num_labels = num_labels
        self.pooling = pooling

        # Encoder only
        self.encoder = AutoModel.from_pretrained(model_name, config=config)

        hidden = config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout)

        # Custom head
        if head_type == "linear":
            self.classifier = nn.Linear(hidden, num_labels)
        elif head_type == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(hidden_dropout),
                nn.Linear(hidden, num_labels),
            )
        else:
            raise ValueError("head_type must be 'linear' or 'mlp'")

        # Freeze base encoder if requested
        if freeze_base:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _pool(self, last_hidden_state, attention_mask=None):
        # last_hidden_state: [B, T, H]
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]  # CLS token
        elif self.pooling == "mean":
            # mean pooling over non-pad tokens
            if attention_mask is None:
                return last_hidden_state.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
            summed = (last_hidden_state * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            return summed / denom
        else:
            raise ValueError("pooling must be 'cls' or 'mean'")

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        pooled = self._pool(enc_out.last_hidden_state, attention_mask)
        x = self.dropout(pooled)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())

        # Keeps your trainer code unchanged: outputs.logits works
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=enc_out.hidden_states,
            attentions=enc_out.attentions,
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

# model = VulDetector(
#     model_name="microsoft/unixcoder-base",
#     num_labels=2,
#     pooling="mean",
#     head_type="mlp",
#     hidden_dropout=0.1
# )
