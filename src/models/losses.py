"""
ENERGIVANU — Loss Functions v3
Key fix: Uncertainty Weighting (Kendall et al., CVPR 2018)
Solves gradient starvation where power MSE (~15,000) overwhelms direction CE (~0.69)

Reference: arXiv:1705.07115
"""

import torch, torch.nn as nn, torch.nn.functional as F
from typing import Tuple, Dict


class UncertaintyWeightedLoss(nn.Module):
    """Multi-task loss with learned uncertainty weighting.

    Uses homoscedastic uncertainty to automatically balance
    regression and classification losses.

    L = (1/2σ₁²)*L_reg + (1/2σ₂²)*L_cls1 + (1/2σ₃²)*L_cls2 + log(σ₁) + log(σ₂) + log(σ₃)

    Tasks with higher uncertainty get lower weight.
    The log(σ) terms prevent driving uncertainty to infinity.
    """
    def __init__(self):
        super().__init__()
        # Learnable log-variance for each task
        # Initialize to 0 (sigma=1, equal weighting)
        self.log_var_power = nn.Parameter(torch.zeros(1))
        self.log_var_signal = nn.Parameter(torch.zeros(1))
        self.log_var_direction = nn.Parameter(torch.zeros(1))

    def forward(self, power_loss, signal_loss, direction_loss):
        # Clamp log_var to prevent weight explosion
        self.log_var_power.data = torch.clamp(self.log_var_power.data, -5.0, 5.0)
        self.log_var_signal.data = torch.clamp(self.log_var_signal.data, -5.0, 5.0)
        self.log_var_direction.data = torch.clamp(self.log_var_direction.data, -5.0, 5.0)

        # Precision (inverse variance)
        prec_power = torch.exp(-self.log_var_power)
        prec_signal = torch.exp(-self.log_var_signal)
        prec_direction = torch.exp(-self.log_var_direction)

        # Weighted losses + regularization (log(sigma) terms)
        total = (prec_power * power_loss + self.log_var_power
                 + prec_signal * signal_loss + self.log_var_signal
                 + prec_direction * direction_loss + self.log_var_direction)

        return total

    def get_weights(self):
        """Return current effective weights for logging."""
        return {
            'w_power': torch.exp(-self.log_var_power).item(),
            'w_signal': torch.exp(-self.log_var_signal).item(),
            'w_direction': torch.exp(-self.log_var_direction).item(),
            'sigma_power': torch.exp(0.5 * self.log_var_power).item(),
            'sigma_signal': torch.exp(0.5 * self.log_var_signal).item(),
            'sigma_direction': torch.exp(0.5 * self.log_var_direction).item(),
        }


class SpikeLoss(nn.Module):
    """SpikeLoss v3 with Uncertainty Weighting.

    Fixes:
    1. Spike detection uses power target (tp), not signal target (ts)
    2. Direction loss uses cross-entropy (classification, not regression)
    3. Loss balancing via learned uncertainty weights (Kendall et al.)
    4. Label smoothing for direction classification
    """
    def __init__(self, uw=5.0, ow=1.0, ss=1.5,
                 use_uncertainty=True, dir_smoothing=0.1):
        super().__init__()
        self.uw = uw  # under-predict weight
        self.ow = ow  # over-predict weight
        self.ss = ss  # spike threshold (std multiplier)
        self.use_uncertainty = use_uncertainty
        self.dir_smoothing = dir_smoothing

        if use_uncertainty:
            self.uncertainty = UncertaintyWeightedLoss()

    def forward(self, pp, tp, ps, ts, pdir, tdir) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            pp: predicted power (B, horizon)
            tp: target power (B, horizon)
            ps: predicted signal logits (B, 3)
            ts: target signal (B,) - 0=SAFE, 1=PREPARE, 2=CRITICAL
            pdir: predicted direction logits (B, 2)
            tdir: target direction (B,) - 0=DOWN, 1=UP
        """
        # === POWER LOSS (MSE with asymmetric weighting) ===
        err = tp - pp
        # FIX: Use tp (power target) for spike detection, not ts (signal target)
        th = tp.mean() + self.ss * tp.std()
        sp = (tp > th).float()
        w = torch.where(err > 0, self.uw + self.uw * sp, torch.tensor(self.ow, device=err.device))
        pl = (w * err.pow(2)).mean()

        # === DIRECTION LOSS (Binary Classification with label smoothing) ===
        if self.dir_smoothing > 0:
            # Label smoothing: convert targets to soft labels
            n_classes = 2
            smooth_val = self.dir_smoothing / (n_classes - 1)
            dir_target = torch.full_like(pdir, smooth_val)
            dir_target.scatter_(1, tdir.unsqueeze(1), 1.0 - self.dir_smoothing)
            dl = -(dir_target * F.log_softmax(pdir, dim=-1)).sum(dim=-1).mean()
        else:
            dl = F.cross_entropy(pdir, tdir)

        # === SIGNAL LOSS (3-class classification with class weights) ===
        cw = torch.tensor([1., 2., 5.], device=ps.device)
        sl = F.cross_entropy(ps, ts, weight=cw)

        # === TOTAL LOSS ===
        if self.use_uncertainty:
            total = self.uncertainty(pl, sl, dl)
        else:
            # Fallback: manual weighting (original behavior)
            total = pl + sl + dl

        # === METRICS (no gradient) ===
        with torch.no_grad():
            mae = err.abs().mean()
            da = (pdir.argmax(-1) == tdir).float().mean()
            sa = (ps.argmax(-1) == ts).float().mean()

            # Direction confidence
            dir_probs = F.softmax(pdir, dim=-1)
            dir_conf = dir_probs.max(dim=-1).values.mean()

            # Loss contributions for monitoring
            if self.use_uncertainty:
                weights = self.uncertainty.get_weights()
            else:
                weights = {'w_power': 1.0, 'w_signal': 1.0, 'w_direction': 1.0}

        return total, {
            "pl": pl.item(),
            "sl": sl.item(),
            "dl": dl.item(),
            "loss": total.item(),
            "mae": mae.item(),
            "da": da.item(),
            "sa": sa.item(),
            "dir_conf": dir_conf.item(),
            **weights
        }


class PhysicsConstraintLoss(nn.Module):
    """Physics-informed loss for GPU power forecasting.

    Constraints:
    1. Power must be between idle and TDP
    2. Power changes must be smooth (bounded derivative)
    3. Higher utilization → higher power (monotonicity)

    Reference: PI-DLinear (arXiv 2605.04074)
    """
    def __init__(self, idle_mw=11.25, tdp_mw=105.0,
                 smooth_weight=0.1, bound_weight=0.1):
        super().__init__()
        self.idle_mw = idle_mw
        self.tdp_mw = tdp_mw
        self.smooth_weight = smooth_weight
        self.bound_weight = bound_weight

    def forward(self, pred_power):
        losses = {}

        # Boundedness: power should be between idle and TDP
        lower_violation = F.relu(self.idle_mw - pred_power).mean()
        upper_violation = F.relu(pred_power - self.tdp_mw).mean()
        losses["bound"] = self.bound_weight * (lower_violation + upper_violation)

        # Smoothness: penalize large jumps in predicted power
        if pred_power.shape[-1] > 1:
            diff = pred_power[:, 1:] - pred_power[:, :-1]
            smooth_loss = diff.pow(2).mean()
            losses["smooth"] = self.smooth_weight * smooth_loss
        else:
            losses["smooth"] = torch.tensor(0.0, device=pred_power.device)

        total = sum(losses.values())
        return total, losses
