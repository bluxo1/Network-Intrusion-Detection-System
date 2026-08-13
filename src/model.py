"""PyTorch neural-network architectures for the NIDS.

Two feed-forward classifiers share a small configurable MLP builder:

  * ``BinaryClassifier``     -> 1 logit  (Normal vs Attack, BCEWithLogitsLoss)
  * ``MultiClassClassifier`` -> 5 logits (Normal/DOS/PROBE/R2L/U2R, CrossEntropy)

Both output raw logits; the loss functions apply sigmoid/softmax internally,
and inference code applies the activation explicitly.
"""

from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    """A generic multi-layer perceptron with optional BatchNorm and dropout.

    Args:
        input_dim: number of input features (after encoding + scaling).
        hidden_dims: sizes of the hidden layers, in order.
        dropouts: dropout probability applied after each hidden layer
            (same length as ``hidden_dims``; use 0.0 to disable).
        output_dim: number of output logits.
        use_batchnorm: if True, insert BatchNorm1d after the first linear layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropouts: List[float],
        output_dim: int,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        assert len(hidden_dims) == len(dropouts), "hidden_dims and dropouts differ in length"

        layers: List[nn.Module] = []
        prev = input_dim
        for i, (h, p) in enumerate(zip(hidden_dims, dropouts)):
            layers.append(nn.Linear(prev, h))
            # BatchNorm on the first (widest) layer stabilises training on the
            # imbalanced multi-class problem.
            if use_batchnorm and i == 0:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if p and p > 0.0:
                layers.append(nn.Dropout(p))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


class BinaryClassifier(MLP):
    """Normal-vs-Attack detector. Emits a single logit per sample."""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropouts: List[float]) -> None:
        super().__init__(input_dim, hidden_dims, dropouts, output_dim=1, use_batchnorm=False)


class MultiClassClassifier(MLP):
    """5-class attack-type classifier. Emits ``num_classes`` logits per sample."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropouts: List[float],
        num_classes: int = 5,
    ) -> None:
        super().__init__(input_dim, hidden_dims, dropouts, output_dim=num_classes, use_batchnorm=True)


def build_models(input_dim: int, model_cfg: dict, num_classes: int = 5):
    """Instantiate both models from a configuration dict (``config.yaml['model']``)."""
    binary = BinaryClassifier(
        input_dim=input_dim,
        hidden_dims=model_cfg["binary_hidden"],
        dropouts=model_cfg["binary_dropout"],
    )
    multiclass = MultiClassClassifier(
        input_dim=input_dim,
        hidden_dims=model_cfg["multiclass_hidden"],
        dropouts=model_cfg["multiclass_dropout"],
        num_classes=num_classes,
    )
    return binary, multiclass
