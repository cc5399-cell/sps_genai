"""Training loops, moved from the Module 4 (classifiers) and Module 5 (VAE) notebooks."""
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from helper_lib.utils import save_checkpoint


def train_model(model, data_loader, criterion, optimizer, device="cpu", epochs=10,
                checkpoint_dir=None):
    """Train a classifier for the given number of epochs and return it.

    If checkpoint_dir is given, a checkpoint is saved every epoch, and the
    best epoch so far (by training accuracy) is also saved in checkpoint_dir/best.
    """
    model.to(device)
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct, running_total = 0, 0

        progress = tqdm(data_loader, dynamic_ncols=True, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_number, (inputs, labels) in enumerate(progress):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()

            if batch_number % 100 == 99:
                progress.set_postfix({
                    "avg accuracy": f"{running_correct / running_total:.3f}",
                    "avg loss": f"{running_loss / (batch_number + 1):.4f}",
                })

        epoch_loss = running_loss / len(data_loader)
        epoch_accuracy = 100 * running_correct / running_total
        print(f"Epoch {epoch + 1}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.2f}%")

        if checkpoint_dir:
            path = save_checkpoint(model, optimizer, epoch + 1, epoch_loss, epoch_accuracy,
                                   checkpoint_dir)
            print(f"Checkpoint saved: {path}")
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                save_checkpoint(model, optimizer, epoch + 1, epoch_loss, epoch_accuracy,
                                os.path.join(checkpoint_dir, "best"))
                print(f"New best model saved! Accuracy: {epoch_accuracy:.2f}%")

    print("Finished Training")
    return model


def vae_loss_function(recon_x, x, mu, logvar, beta=500):
    """VAE loss from the Module 5 notebook.

    beta * reconstruction error (binary cross-entropy) + KL divergence, which
    pushes the latent distribution towards a standard normal N(0, 1).
    """
    bce = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return beta * bce + kld


def train_vae_model(model, data_loader, criterion, optimizer, device="cpu", epochs=10):
    """Train a VAE for the given number of epochs and return it.

    criterion must take (reconstruction, inputs, mu, logvar), e.g. vae_loss_function.
    Labels are ignored: the VAE learns to reconstruct its own input.
    """
    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        progress = tqdm(data_loader, dynamic_ncols=True, desc=f"Epoch {epoch + 1}/{epochs}")
        for inputs, _ in progress:
            inputs = inputs.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(inputs)
            loss = criterion(recon, inputs, mu, logvar)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item() / inputs.size(0):.1f}")

        avg_loss = running_loss / len(data_loader.dataset)
        print(f"Epoch {epoch + 1}: average loss per image = {avg_loss:.1f}")

    print("Finished Training")
    return model


def train_gan(model, data_loader, criterion=None, optimizer=None, device="cpu", epochs=10,
              n_critic=5, clip_value=0.01, lr=5e-5):
    """Train a Wasserstein GAN (Module 6 notebook) and return it.

    criterion: unused; the WGAN losses are computed directly from the critic scores.
    optimizer: (generator_optimizer, critic_optimizer); if None, RMSprop(lr) is created for each.
    The critic is updated n_critic times per generator update, and its weights are clipped
    to [-clip_value, clip_value] to enforce the Lipschitz condition.
    """
    gen, critic = model.generator.to(device), model.critic.to(device)
    if optimizer is None:
        optimizer = (torch.optim.RMSprop(gen.parameters(), lr=lr),
                     torch.optim.RMSprop(critic.parameters(), lr=lr))
    opt_gen, opt_critic = optimizer
    model.train()

    for epoch in range(epochs):
        progress = tqdm(data_loader, dynamic_ncols=True, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_number, (real, _) in enumerate(progress):
            real = real.to(device)
            batch_size = real.size(0)

            # Train the critic: loss = -(E[D(real)] - E[D(fake)])
            for _ in range(n_critic):
                noise = torch.randn(batch_size, model.z_dim, device=device)
                fake = gen(noise).detach()
                loss_critic = -(critic(real).mean() - critic(fake).mean())
                critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()
                for p in critic.parameters():  # weight clipping
                    p.data.clamp_(-clip_value, clip_value)

            # Train the generator: loss = -E[D(G(z))]
            noise = torch.randn(batch_size, model.z_dim, device=device)
            loss_gen = -critic(gen(noise)).mean()
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_number % 100 == 0:
                progress.set_postfix({"D loss": f"{loss_critic.item():.4f}",
                                      "G loss": f"{loss_gen.item():.4f}"})

        print(f"Epoch {epoch + 1}: D loss={loss_critic.item():.4f}, G loss={loss_gen.item():.4f}")

    print("Finished Training")
    return model


def train_standard_gan(model, data_loader, criterion=None, optimizer=None, device="cpu", epochs=10, lr=2e-4):
    """Train a standard GAN with binary cross-entropy (Module 6 GAN losses) and return it.

    Discriminator loss: -[log D(x) + log(1 - D(G(z)))]
    Generator loss:     -log D(G(z))
    criterion: defaults to BCEWithLogitsLoss (applies the sigmoid to the discriminator's output).
    optimizer: (generator_optimizer, discriminator_optimizer); defaults to Adam(lr, betas=(0.5, 0.999)).
    """
    gen, disc = model.generator.to(device), model.discriminator.to(device)
    criterion = criterion or torch.nn.BCEWithLogitsLoss()
    if optimizer is None:
        optimizer = (torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999)),
                     torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999)))
    opt_gen, opt_disc = optimizer
    model.train()

    for epoch in range(epochs):
        progress = tqdm(data_loader, dynamic_ncols=True, desc=f"Epoch {epoch + 1}/{epochs}")
        for real, _ in progress:
            real = real.to(device)
            batch_size = real.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Train the discriminator: real images -> 1, generated images -> 0
            fake = gen(torch.randn(batch_size, model.z_dim, device=device))
            loss_disc = criterion(disc(real), real_labels) + criterion(disc(fake.detach()), fake_labels)
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train the generator: make the discriminator call generated images real
            loss_gen = criterion(disc(fake), real_labels)
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            progress.set_postfix({"D loss": f"{loss_disc.item():.4f}", "G loss": f"{loss_gen.item():.4f}"})

        print(f"Epoch {epoch + 1}: D loss={loss_disc.item():.4f}, G loss={loss_gen.item():.4f}")

    print("Finished Training")
    return model



def train_rnn_model(model, data_loader, criterion, optimizer, device="cpu", epochs=10):
    """Train a next-word language model (Module 6 RNN notebook) and return it."""
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        progress = tqdm(data_loader, dynamic_ncols=True, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_number, (inputs, targets) in enumerate(progress):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            # Flatten (batch, seq, vocab) so every position is one classification problem
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_number % 100 == 0:
                progress.set_postfix({"avg loss": f"{total_loss / (batch_number + 1):.4f}"})

        print(f"Epoch {epoch + 1}: average loss = {total_loss / len(data_loader):.4f}")

    print("Finished Training")
    return model



def train_diffusion(model, data_loader, criterion=None, optimizer=None, device="cpu", epochs=10):
    """Train a DiffusionModel (Module 8 notebook) and return it.

    criterion: defaults to L1Loss between predicted and true noise.
    optimizer: defaults to AdamW(lr=1e-3, weight_decay=1e-4) on the main network.
    """
    criterion = criterion or torch.nn.L1Loss()
    optimizer = optimizer or torch.optim.AdamW(model.network.parameters(), lr=1e-3, weight_decay=1e-4)
    model.to(device)

    # Per-channel mean and std of the training images, used to normalize them
    total, mean, std = 0, 0.0, 0.0
    for images, _ in data_loader:
        n = images.size(0)
        flat = images.view(n, images.size(1), -1)
        mean = mean + flat.mean(dim=(0, 2)) * n
        std = std + flat.std(dim=(0, 2)) * n
        total += n
    model.set_normalizer((mean / total).to(device), (std / total).to(device))

    for epoch in range(epochs):
        model.train()
        losses = []
        progress = tqdm(data_loader, dynamic_ncols=True, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, _ in progress:
            loss = model.train_step(images.to(device), optimizer, criterion)
            losses.append(loss)
            progress.set_postfix(loss=f"{loss:.4f}")
        print(f"Epoch {epoch + 1}: average loss = {sum(losses) / len(losses):.4f}")

    print("Finished Training")
    return model


# ---------- Assignment 4: Energy-based model training (Module 7 notebook) ----------

class ReplayBuffer:
    """Recent Langevin samples, so each step continues from earlier ones (Module 7 notebook)."""

    def __init__(self, model, image_shape, device, size=8192):
        self.model = model
        self.image_shape = tuple(image_shape)
        self.device = device
        self.size = size
        self.examples = [torch.rand((1, *self.image_shape), device=device) * 2 - 1 for _ in range(128)]

    def sample_new_exmps(self, batch_size, steps, step_size, noise):
        # About 5% fresh random images, the rest continued from the buffer
        n_new = int((torch.rand(batch_size) < 0.05).sum())
        new_rand_imgs = torch.rand((n_new, *self.image_shape), device=self.device) * 2 - 1
        idx = torch.randint(len(self.examples), (batch_size - n_new,))
        old_imgs = torch.cat([self.examples[i] for i in idx], dim=0)

        new_imgs = self.model.sample(torch.cat([new_rand_imgs, old_imgs], dim=0), steps, step_size, noise)

        self.examples = (list(torch.split(new_imgs, 1, dim=0)) + self.examples)[:self.size]
        return new_imgs


def train_ebm(model, data_loader, optimizer=None, device="cpu", epochs=10,
              alpha=0.1, steps=60, step_size=10.0, noise=0.005):
    """Train an EnergyModel with contrastive divergence (Module 7 notebook) and return it.

    Expects images in [0, 1] (they are rescaled to [-1, 1] here).
    Loss = E(real) - E(fake) + alpha * (E(real)^2 + E(fake)^2): push energy down on real
    images and up on Langevin samples, with a regularizer that keeps energies small.
    """
    model.to(device)
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))
    buffer = ReplayBuffer(model, data_loader.dataset[0][0].shape, device)

    for epoch in range(epochs):
        model.train()
        progress = tqdm(data_loader, dynamic_ncols=True, desc=f"Epoch {epoch + 1}/{epochs}")
        for real_imgs, _ in progress:
            real_imgs = real_imgs.to(device) * 2 - 1
            real_imgs = (real_imgs + torch.randn_like(real_imgs) * noise).clamp(-1.0, 1.0)
            fake_imgs = buffer.sample_new_exmps(real_imgs.size(0), steps, step_size, noise)

            out = model(torch.cat([real_imgs, fake_imgs], dim=0))
            real_out, fake_out = torch.split(out, [real_imgs.size(0), fake_imgs.size(0)], dim=0)
            cdiv_loss = real_out.mean() - fake_out.mean()
            reg_loss = alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())
            loss = cdiv_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            progress.set_postfix({"loss": f"{loss.item():.4f}", "real": f"{real_out.mean().item():.3f}",
                                  "fake": f"{fake_out.mean().item():.3f}"})

        print(f"Epoch {epoch + 1}: loss={loss.item():.4f}, "
              f"E(real)={real_out.mean().item():.3f}, E(fake)={fake_out.mean().item():.3f}")

    print("Finished Training")
    return model
