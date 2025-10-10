import torch
import torch.nn as nn

class CombinedPlayerModel(nn.Module):
    def __init__(self, seq_input_dim, season_input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(seq_input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc_season = nn.Sequential(
            nn.Linear(season_input_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_combined = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, seq_x, season_x):
        _, (hn, _) = self.lstm(seq_x)
        lstm_feat = hn[-1]
        season_feat = self.fc_season(season_x)
        combined = torch.cat([lstm_feat, season_feat], dim=1)
        out = self.fc_combined(combined)
        return out