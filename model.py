import torch.nn as nn


class EyeStateModel(nn.Module):
    """Binary classifier: open vs closed eye. Input (N,1,24,24), output (N,1) logits."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),    # -> 32x11x11
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),   # -> 64x4x4
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2),  # -> 128x1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
