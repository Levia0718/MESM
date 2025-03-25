import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Encoder(nn.Module):
    def __init__(self, input_dim, input_length):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=1, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm1d(1)
        self.biGRU = nn.GRU(1, 1, bidirectional=True, batch_first=True, num_layers=1)
        self.maxpool1d = nn.MaxPool1d(3, stride=3)
        self.global_avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(math.floor(input_length / 3), 640)
        self.mean_layer = nn.Linear(640, 1024)
        self.var_layer = nn.Linear(640, 1024)

    def forward(self, x):
        encoder_output = self.get_encoder_output(x)
        mean = self.mean_layer(encoder_output)
        log_var = self.var_layer(encoder_output)
        return mean, log_var, encoder_output

    def get_encoder_output(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = self.bn1(x)
        x = self.maxpool1d(x)
        x = x.transpose(1, 2)
        x, _ = self.biGRU(x)
        x = self.global_avgpool1d(x)
        x = x.squeeze()
        x = self.fc1(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_length, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 1024)
        self.fc_reconstruct = nn.Linear(1024, output_length * output_dim)
        self.output_dim = output_dim
        self.output_length = output_length

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = self.fc_reconstruct(x)
        x = x.view(-1, self.output_length, self.output_dim)
        return x


class SVAE(nn.Module):
    def __init__(self, input_length, input_dim, latent_dim=1024):
        super(SVAE, self).__init__()
        self.encoder = Encoder(input_dim, input_length)
        self.decoder = Decoder(latent_dim, input_length, input_dim)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, log_var, encoder_output = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mean, log_var, encoder_output

    def compute_loss(self, x, x_reconstructed, mean, log_var):
        recon_loss = F.mse_loss(x_reconstructed, x, reduction='mean')
        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return recon_loss + kl_div / x.size(0)
