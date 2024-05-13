# %%
# Training a TMS
import sys
import pathlib

project_dir = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_dir))
from tms import Model, Config

config = Config(n_instances=8, groups=[2, 2], n_hidden=2)
model = Model(config)
# model.generate_batch_with_groups(4)
# %%

x = model.generate_batch(32)
x1 = model.decode(model.encode(x))
x2 = model(x)

print((x1 - x2).norm())


# %%
Model.generate_batch = Model.generate_batch_with_groups
model.optimize(steps=10_000)

# %%
import numpy as np
from utils import plot_features_in_2d

flat_groups = []
for i, group_size in enumerate(model.cfg.groups):
    flat_groups.extend([0.9 * i] * group_size)
flat_groups = np.array(flat_groups).reshape(1, -1)
flat_groups = np.repeat(flat_groups, model.cfg.n_instances, axis=0)
print(flat_groups)

# colors = np.zeros((config.n_instances, config.n_features))
# for c in colors:
#     c = np.arange(config.n_features) + 1

plot_features_in_2d(
    model.W.detach(),
    colors=flat_groups,
    title=f"Superposition: {model.cfg.n_features} features represented in 2D space",
    # subplot_titles=[f"1 - S = {i:.3f}" for i in feature_probability.squeeze()],
)

# %%
# Now, we train SAEs

# %%

import torch
from smol_sae.base import Config
from smol_sae.vanilla import VanillaSAE
from smol_sae.utils import get_splits
from transformer_lens import HookedTransformer

# %%
import torch.nn as nn
from einops import einsum
from collections import namedtuple

Loss = namedtuple("Loss", ["reconstruction", "sparsity", "auxiliary"])


class SAE(nn.Module):
    """
    Base class for all Sparse Auto Encoders.
    Provides a common interface for training and evaluation.
    """

    def __init__(
        self,
        d_model: int,
        expansion: int,
        sparsities: tuple[float],
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_hidden = expansion * self.d_model
        self.n_instances = len(sparsities)

        self.steps_not_active = torch.zeros(self.n_instances, self.d_hidden)
        self.sparsities = torch.tensor(sparsities).to(device)
        self.step = 0

        W_dec = torch.randn(
            self.n_instances, self.d_hidden, self.d_model, device=device
        )
        self._W_dec = nn.Parameter(W_dec)

        W_enc = W_dec.mT.clone().to(device)
        self.W_enc = nn.Parameter(W_enc)

        self.b_enc = nn.Parameter(
            torch.zeros(self.n_instances, self.d_hidden, device=device)
        )
        self.b_dec = nn.Parameter(
            torch.zeros(self.n_instances, self.d_model, device=device)
        )
        self.relu = nn.ReLU()

    @property
    def W_dec(self):
        """Return normalized decoder"""
        # NOTE: Check which dimension to normalize over
        return self._W_dec / self._W_dec.norm(dim=-1, keepdim=True)

    def encode(self, x):
        return self.relu(
            einsum(
                x - self.b_dec,
                self.W_enc,
                "... inst d, inst d hidden -> ... inst hidden",
            )
            + self.b_enc
        )

    def decode(self, h):
        return (
            einsum(h, self.W_dec, "... inst hidden, inst hidden d -> ... inst d")
            + self.b_dec
        )

    def forward(self, x):
        x_hid, *_ = self.encode(x)
        return self.decode(x_hid)

    def loss(self, x, x_hid, x_hat):
        reconstruction = ((x_hat - x) ** 2).mean(0).sum(dim=-1)

        l1_loss = x_hid.abs().mean(dim=0).sum(-1)

        return Loss(
            reconstruction,
            self.sparsities * l1_loss,
            torch.zeros(self.n_instances, device=x.device),
        )


# %%
# Test that SAE works
import torch

device = "cuda"
sae = SAE(d_model=2, expansion=3, sparsities=(0.01, 0.03, 0.1, 0.3, 1.0), device=device)

n_batch = 2
input = torch.randn(n_batch, sae.n_instances, sae.d_model).to(device)
sae.forward(input)


# %%
from einops import repeat

# Sample data from TMS.generate_batch_with_group


def get_sae_data(batch_size: int):
    data = model.generate_batch_with_groups(10_000)

    # Forward pass of TMS, get the hidden layer activations
    x = model.encode(data)
    x = x[:, 0, :]
    print(x.shape)

    # This forms the data for the SAE
    x = repeat(x, "batch d_hidden -> batch n_sae d_hidden", n_sae=sae.n_instances)
    return x


x = get_sae_data(10_000)
x_rec = sae(x)
# print(x_rec.shape)

# Train the SAE on the activations


# %%

from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD


with torch.no_grad():
    train_dataset = TensorDataset(get_sae_data(10_000))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
sae = SAE(d_model=2, expansion=2, sparsities=(0.01, 0.03, 0.1, 0.3, 1.0), device=device)
optimizer = SGD(sae.parameters(), lr=0.01)


def train(sae: SAE, train_loader, n_epochs=10):
    for epoch in range(n_epochs):
        for batch in train_loader:
            x = batch[0].to(device)
            x_hid = sae.encode(x)
            x_hat = sae.decode(x_hid)

            losses = sae.loss(x, x_hid, x_hat)
            loss = (losses.reconstruction + losses.sparsity + losses.auxiliary).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(losses.reconstruction.mean().item())


train(sae, train_loader)

# %%

# Combine the W_dec of SAE and model

first_model_W = model.W[0].detach()
# repeat
first_model_W = repeat(
    first_model_W,
    "d_model_hid d_data -> n_sae d_model_hid d_data",
    n_sae=sae.n_instances,
)
print(first_model_W.shape)
print(sae.W_dec.shape)

combined = torch.cat([first_model_W, sae.W_dec.transpose(1, 2)], dim=-1)

model_colors = np.zeros((sae.n_instances, 4)) + 0.2
sae_colors = np.zeros((sae.n_instances, sae.d_hidden)) + 1.8
combined_colors = np.concatenate([model_colors, sae_colors], axis=-1)
print(combined_colors.shape)

# print(model_colors)
# print(sae_colors)

plot_features_in_2d(
    combined,
    # sae.W_dec.detach(),
    colors=combined_colors,
    # colors=np.concatenate([np.zeros_like(flat_groups[:5]), sae_colors], axis=-1),
    title=f"Superposition: {model.cfg.n_features} features represented in 2D space",
    # subplot_titles=[f"1 - S = {i:.3f}" for i in feature_probability.squeeze()],
)

# %%
print(sae.W_dec.shape)
print(sae.W_dec.norm(dim=-1))

# %%
