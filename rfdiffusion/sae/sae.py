from torch import nn

class SAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__(self)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.GELU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed