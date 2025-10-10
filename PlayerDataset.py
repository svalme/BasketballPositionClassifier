import numpy as np
import torch
from torch.utils.data import Dataset


class CombinedPlayerDataset(Dataset):
    def __init__(self, df_seq, df_season, seq_features, season_features, label_col, seq_len=82):
        self.players = df_seq['PLAYER_ID'].unique()
        self.seq_features = seq_features
        self.season_features = season_features
        self.seq_len = seq_len

        self.labels = []
        self.sequences = []
        self.season_feats = []

        for pid in self.players:
            # Sequence
            player_games = df_seq[df_seq['PLAYER_ID']==pid].sort_values('GAME_NUM')[seq_features].values
            if len(player_games) < seq_len:
                padding = np.zeros((seq_len - len(player_games), len(seq_features)))
                player_games = np.vstack([player_games, padding])
            else:
                player_games = player_games[:seq_len]
            self.sequences.append(player_games)

            # Season features
            season_row = df_season[df_season['PLAYER_ID']==pid][season_features].values
            self.season_feats.append(season_row.squeeze())

            # Label
            label = df_season[df_season['PLAYER_ID']==pid][label_col].iloc[0]
            self.labels.append(label)

        self.sequences = torch.tensor(self.sequences, dtype=torch.float32)
        self.season_feats = torch.tensor(self.season_feats, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.season_feats[idx], self.labels[idx]