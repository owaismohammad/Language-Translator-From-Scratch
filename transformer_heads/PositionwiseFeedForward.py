import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model:int, hidden_unit:int, drop_prob:float):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden_unit),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_unit, d_model),
        )
    def forward(self,x):
        return self.fc(x)

