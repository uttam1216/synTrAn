import torch.nn as nn
import torch


class UserPredictionModel(nn.Module):
    def __init__(self, num_locations, embedding_dim=32, hidden_dim=64):
        super(UserPredictionModel, self).__init__()
        self.timestamp_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU()
        )
        self.location_embedding = nn.Sequential(
            nn.Embedding(num_locations, embedding_dim),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, timestamps, location_ids):
        timestamp_embeddings = self.timestamp_embedding(timestamps)
        location_embeddings = self.location_embedding(location_ids)
        x = torch.cat([timestamp_embeddings, location_embeddings], dim=1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x
