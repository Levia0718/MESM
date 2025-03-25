import torch.nn as nn


class FAE(nn.Module):

    def __init__(self, latent_dim, input_features):
        super(FAE, self).__init__()
        self.input_features = input_features

        self.fuse_in = nn.Sequential(
            nn.Linear(input_features, input_features // 2),
            nn.Tanh(),
            nn.Linear(input_features // 2, latent_dim),
            nn.ReLU()
        )
        self.fuse_out = nn.Sequential(
            nn.Linear(latent_dim, input_features // 2),
            nn.ReLU(),
            nn.Linear(input_features // 2, input_features)
        )
        self.criterion = nn.MSELoss()

    def encode(self, z):
        compressed_z = self.fuse_in(z)
        return compressed_z

    def decode(self, compressed_z):
        z = self.fuse_out(compressed_z)
        return z
