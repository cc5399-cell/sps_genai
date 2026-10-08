"""Model definitions, moved from the Module 4 (FCNN, CNN) and Module 5 (VAE) notebooks."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Module 4: classifiers for CIFAR-10 ----------

class FCNN(nn.Module):
    """Fully connected network: flattens each 3x32x32 image into a 3072-long vector."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SimpleCNN(nn.Module):
    """First CNN from Practical 3: two conv + pool blocks, then two fully connected layers."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class EnhancedCNN(nn.Module):
    """Enhanced CNN from Practical 3: four conv layers with BatchNorm, plus Dropout."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ---------- Module 5: Variational Autoencoder for Fashion-MNIST ----------

class VariationalEncoder(nn.Module):
    """Maps a 1x32x32 image to the mean and log-variance of a normal distribution."""

    def __init__(self, latent_dim=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)    # 32 -> 16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)   # 16 -> 8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 8 -> 4
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """Maps a latent vector back to a 1x32x32 image (transpose convolutions)."""

    def __init__(self, latent_dim=2):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.convtrans1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtrans2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtrans3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 4, 4)
        x = F.relu(self.convtrans1(x))       # 4 -> 8
        x = F.relu(self.convtrans2(x))       # 8 -> 16
        x = torch.sigmoid(self.convtrans3(x))  # 16 -> 32, pixel values in [0, 1]
        return x


class VAE(nn.Module):
    """Variational Autoencoder: encode to a distribution, sample from it, then decode."""

    def __init__(self, encoder=None, decoder=None, latent_dim=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder if encoder is not None else VariationalEncoder(latent_dim)
        self.decoder = decoder if decoder is not None else Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar


MODELS = {
    "FCNN": FCNN,
    "CNN": SimpleCNN,
    "EnhancedCNN": EnhancedCNN,
    "VAE": VAE,
}


def get_model(model_name):
    """Return a new model by name: one of FCNN, CNN, EnhancedCNN, VAE (not case-sensitive)."""
    for name, model_class in MODELS.items():
        if name.lower() == model_name.lower():
            return model_class()
    raise ValueError(f"Unknown model '{model_name}'. Choose one of: {', '.join(MODELS)}")


# ---------- Assignment 2: CNN built to the given specification ----------

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class AssignmentCNN(nn.Module):
    """CNN from the Assignment 2 specification.

    Shape trace for a 64x64x3 input:
      Conv2d(3->16, 3x3, stride 1, pad 1) + ReLU   -> 64x64x16
      MaxPool2d(2x2, stride 2)                     -> 32x32x16
      Conv2d(16->32, 3x3, stride 1, pad 1) + ReLU  -> 32x32x32
      MaxPool2d(2x2, stride 2)                     -> 16x16x32
      Flatten                                      -> 8192
      Linear(8192 -> 100) + ReLU                   -> 100
      Linear(100 -> 10)                            -> 10 class scores
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


MODELS["AssignmentCNN"] = AssignmentCNN
