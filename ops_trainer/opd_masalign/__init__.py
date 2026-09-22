"""OPD-MASAlign: on-policy self-distillation training on labeled critical steps.

See docs/DATA_FORMAT.md for the training-time hand-off schema
(CriticalStepRecord) that opd_trainer consumes directly. There is no
tree/graph/risk-propagation pipeline in this package -- critical steps are
located and outcome-labeled upstream (e.g. against a collaborator's raw
rollout logs) and handed off already in this format.
"""
from .schema import validate_critical_step_record
from .self_teacher import build_reprompt_text, build_teacher_messages

try:
    from .losses import compute_opd_loss, ema_update  # requires torch
except ImportError:
    compute_opd_loss = ema_update = None  # torch not installed; schema/self_teacher still usable

__all__ = [
    "validate_critical_step_record",
    "build_reprompt_text",
    "build_teacher_messages",
    "compute_opd_loss",
    "ema_update",
]
