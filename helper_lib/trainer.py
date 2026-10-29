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
