"""
Phase 2.1 — Model definition.

Small MLP for binary (normal-vs-attack) classification over the union
feature space produced by data/prepare_data.py. Kept small/fast since it
retrains locally every FL round across many clients and rounds.
"""
import torch.nn as nn


class IDSNet(nn.Module):
    def __init__(self, input_dim: int, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits, BCEWithLogitsLoss expects raw logits
