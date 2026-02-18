"""Loss functions: Focal Loss and combined multi-head loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """Binary Focal Loss for imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: weighting factor for positive class. Default 0.25.
        gamma: focusing parameter. Default 2.0.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits: (N,) raw logits (before sigmoid).
            targets: (N,) binary labels {0, 1}.

        Returns:
            loss: scalar.
        """
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = (focal_weight * ce).mean()
        return loss


class CombinedLoss(nn.Module):
    """Combined loss for multi-head fall detection model.

    Heads:
        - state: 3-class cross-entropy
        - impact: focal BCE
        - prediction: focal BCE

    Args:
        state_weight: weight for state CE loss.
        impact_weight: weight for impact focal loss.
        pred_weight: weight for prediction focal loss.
        focal_alpha: alpha for focal loss.
        focal_gamma: gamma for focal loss.
    """

    def __init__(self, state_weight: float = 1.0,
                 impact_weight: float = 2.0,
                 pred_weight: float = 1.0,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        super().__init__()
        self.state_weight = state_weight
        self.impact_weight = impact_weight
        self.pred_weight = pred_weight

        self.state_ce = nn.CrossEntropyLoss()
        self.impact_focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.pred_focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, state_logits: Tensor, impact_logits: Tensor,
                pred_logits: Tensor, state_targets: Tensor,
                impact_targets: Tensor, pred_targets: Tensor
                ) -> dict:
        """
        Args:
            state_logits: (B, 3)
            impact_logits: (B,) or (B, 1)
            pred_logits: (B,) or (B, 1)
            state_targets: (B,) long
            impact_targets: (B,) float
            pred_targets: (B,) float

        Returns:
            dict with 'loss_state', 'loss_impact', 'loss_pred', 'loss_total'.
        """
        impact_logits = impact_logits.squeeze(-1)
        pred_logits = pred_logits.squeeze(-1)

        loss_state = self.state_ce(state_logits, state_targets)
        loss_impact = self.impact_focal(impact_logits, impact_targets)
        loss_pred = self.pred_focal(pred_logits, pred_targets)

        loss_total = (self.state_weight * loss_state +
                      self.impact_weight * loss_impact +
                      self.pred_weight * loss_pred)

        return {
            'loss_state': loss_state,
            'loss_impact': loss_impact,
            'loss_pred': loss_pred,
            'loss_total': loss_total,
        }
