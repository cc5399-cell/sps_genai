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


# ---------- Module 6: Wasserstein GAN (from the Module 6 GAN notebook) ----------

class Critic(nn.Module):
    """WGAN critic: 3x64x64 image -> one unbounded realness score."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),      # 64 -> 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),    # 32 -> 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),   # 16 -> 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),   # 8 -> 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),                                                           # 4 -> 1
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    """WGAN generator: z_dim noise vector -> 3x64x64 image in [-1, 1]."""

    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False), nn.BatchNorm2d(512, momentum=0.9), nn.ReLU(True),  # 1 -> 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256, momentum=0.9), nn.ReLU(True),    # 4 -> 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128, momentum=0.9), nn.ReLU(True),    # 8 -> 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64, momentum=0.9), nn.ReLU(True),      # 16 -> 32
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh(),                                                          # 32 -> 64
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), self.z_dim, 1, 1))


class GAN(nn.Module):
    """Holds the generator and critic together so the GAN works with get_model()."""

    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim)
        self.critic = Critic()

    def forward(self, z):
        return self.generator(z)


MODELS["GAN"] = GAN


# ---------- Assignment 3: GAN built to the given specification (MNIST, 1x28x28) ----------

class MNISTGenerator(nn.Module):
    """Noise (batch, 100) -> FC to 7x7x128 -> ConvT 128->64 (14x14) + BN + ReLU -> ConvT 64->1 (28x28) + Tanh."""

    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 7x7 -> 14x14
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)    # 14x14 -> 28x28

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 7, 7)       # reshape to 128 x 7 x 7
        x = F.relu(self.bn1(self.deconv1(x)))
        return torch.tanh(self.deconv2(x))       # pixels in [-1, 1]


class MNISTDiscriminator(nn.Module):
    """Image (1x28x28) -> Conv 1->64 (14x14) + LeakyReLU -> Conv 64->128 (7x7) + BN + LeakyReLU -> Flatten -> Linear -> 1."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)    # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 14x14 -> 7x7
        self.bn2 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        return self.fc(self.flatten(x))  # logit; sigmoid(logit) = probability the image is real


class MNISTGAN(nn.Module):
    """Holds the generator and discriminator together so the GAN works with get_model()."""

    def __init__(self, z_dim=100):
        super().__init__()
        self.z_dim = z_dim
        self.generator = MNISTGenerator(z_dim)
        self.discriminator = MNISTDiscriminator()

    def forward(self, z):
        return self.generator(z)


MODELS["MNISTGAN"] = MNISTGAN



# ---------- Module 7: LSTM text generator (from the Module 6 RNN notebook) ----------

class LSTMModel(nn.Module):
    """Word-level language model: embedding -> LSTM -> linear layer scoring every vocabulary word."""

    def __init__(self, vocab_size=10000, embedding_dim=100, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        return self.fc(x), hidden


MODELS["LSTM"] = LSTMModel



# ---------- Module 8: Diffusion model (from the Module 8 Diffusion notebook) ----------

import copy
import math


def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    """Map diffusion times in [0, 1] to (noise_rates, signal_rates), with noise^2 + signal^2 = 1."""
    start_angle = math.acos(max_signal_rate)
    end_angle = math.acos(min_signal_rate)
    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
    return torch.sin(diffusion_angles), torch.cos(diffusion_angles)


class SinusoidalEmbedding(nn.Module):
    """Encode the noise variance as sines and cosines at 16 frequencies (32 numbers)."""

    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):
        # x: (B, 1, 1, 1) -> (B, 1, 1, 2 * num_frequencies)
        x = x.expand(-1, 1, 1, self.num_frequencies)
        return torch.cat([torch.sin(self.angular_speeds * x), torch.cos(self.angular_speeds * x)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.proj(x)
        x = F.silu(self.conv1(x))  # swish activation
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    """Residual blocks (saving each output as a skip connection), then halve the resolution."""

    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return self.pool(x)


class UpBlock(nn.Module):
    """Double the resolution, then residual blocks that each consume one skip connection."""

    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        for block in self.blocks:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    """Predicts the noise in a noisy image, given the image and its noise variance."""

    def __init__(self, image_size=64, num_channels=3, embedding_dim=32):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.embedding = SinusoidalEmbedding(num_frequencies=embedding_dim // 2)

        self.down1 = DownBlock(32, in_channels=32 + embedding_dim, block_depth=2)
        self.down2 = DownBlock(64, in_channels=32, block_depth=2)
        self.down3 = DownBlock(96, in_channels=64, block_depth=2)
        self.mid1 = ResidualBlock(in_channels=96, out_channels=128)
        self.mid2 = ResidualBlock(in_channels=128, out_channels=128)
        self.up1 = UpBlock(96, in_channels=128, block_depth=2)
        self.up2 = UpBlock(64, in_channels=96, block_depth=2)
        self.up3 = UpBlock(32, in_channels=64, block_depth=2)

        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        noise_emb = self.embedding(noise_variances).permute(0, 3, 1, 2)  # (B, 32, 1, 1)
        noise_emb = F.interpolate(noise_emb, size=(self.image_size, self.image_size), mode="nearest")
        x = torch.cat([x, noise_emb], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)
        x = self.mid1(x)
        x = self.mid2(x)
        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)
        return self.final(x)


class DiffusionModel(nn.Module):
    """Denoising diffusion model: a UNet plus an EMA copy used for generation."""

    def __init__(self, image_size=64, num_channels=3, schedule_fn=offset_cosine_diffusion_schedule, ema_decay=0.8):
        super().__init__()
        self.network = UNet(image_size, num_channels)
        self.ema_network = copy.deepcopy(self.network)
        self.ema_decay = ema_decay
        self.schedule_fn = schedule_fn
        # Per-channel normalization statistics, saved with the weights
        self.register_buffer("normalizer_mean", torch.zeros(1, num_channels, 1, 1))
        self.register_buffer("normalizer_std", torch.ones(1, num_channels, 1, 1))

    def set_normalizer(self, mean, std):
        self.normalizer_mean.copy_(mean.reshape_as(self.normalizer_mean))
        self.normalizer_std.copy_(std.reshape_as(self.normalizer_std))

    def denormalize(self, x):
        return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # Main network while training, EMA network for inference
        network = self.network if training else self.ema_network
        pred_noises = network(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        for step in range(diffusion_steps):
            t = torch.ones((initial_noise.shape[0], 1, 1, 1), device=initial_noise.device) * (1 - step * step_size)
            noise_rates, signal_rates = self.schedule_fn(t)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)
            next_noise_rates, next_signal_rates = self.schedule_fn(t - step_size)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        return pred_images

    def generate(self, num_images, diffusion_steps):
        """Start from pure Gaussian noise and denoise it step by step into images in [0, 1]."""
        size = self.network.image_size
        initial_noise = torch.randn((num_images, self.network.num_channels, size, size),
                                    device=self.normalizer_mean.device)
        with torch.no_grad():
            return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))

    def train_step(self, images, optimizer, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)

        # Corrupt each image with a random amount of noise
        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        # Train the network to predict the noise that was added
        pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
        loss = loss_fn(pred_noises, noises)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the EMA copy, which generate() uses
        with torch.no_grad():
            for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_param.copy_(self.ema_decay * ema_param + (1.0 - self.ema_decay) * param)

        return loss.item()


MODELS["Diffusion"] = DiffusionModel
