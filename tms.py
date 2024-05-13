import torch as t
import torch
from eindex import eindex
import torch.nn as nn
import einops
from einops import rearrange
import torch.nn.functional as F

from tqdm import tqdm
from dataclasses import dataclass, field
from jaxtyping import Float
from torch import Tensor
from typing import Optional, Union, Callable

t.manual_seed(2)

W = t.randn(2, 5)
W_normed = W / W.norm(dim=0, keepdim=True)
device = "cuda"


def constant_lr(*_):
    return 1.0


@dataclass
class Config:
    # We optimize n_instances models in a single training loop to let us sweep over
    # sparsity or importance curves  efficiently. You should treat `n_instances` as
    # kinda like a batch dimension, but one which is built into our training setup.
    n_instances: int
    # n_features: int = 5
    groups: list[int] = field(default_factory=lambda: [2, 2])
    n_hidden: int = 2

    @property
    def n_features(self) -> int:
        return sum(self.groups)


# Toy model of superposition
class Model(nn.Module):
    W: Float[Tensor, "n_instances n_hidden n_features"]
    b_final: Float[Tensor, "n_instances n_features"]

    def __init__(
        self,
        cfg: Config,
        feature_probability: Optional[Union[float, Tensor]] = None,
        importance: Optional[Union[float, Tensor]] = None,
        device=device,
    ):
        super().__init__()
        self.cfg = cfg

        n_features = sum(cfg.groups)
        self.n_features = n_features

        if feature_probability is None:
            feature_probability = t.ones(())
        if isinstance(feature_probability, float):
            feature_probability = t.tensor(feature_probability)
        self.feature_probability = feature_probability.to(device).broadcast_to(
            (cfg.n_instances, n_features)
        )
        if importance is None:
            importance = t.ones(())
        if isinstance(importance, float):
            importance = t.tensor(importance)
        self.importance = importance.to(device).broadcast_to(
            (cfg.n_instances, n_features)
        )

        self.W = nn.Parameter(
            nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_hidden, n_features)))
        )
        self.b_final = nn.Parameter(t.zeros((cfg.n_instances, n_features)))
        self.to(device)

    def encode(
        self, features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances hidden"]:
        return einops.einsum(
            features,
            self.W,
            "... instances features, instances hidden features -> ... instances hidden",
        )

    def decode(
        self, hidden: Float[Tensor, "... instances hidden"]
    ) -> Float[Tensor, "... instances features"]:
        z = einops.einsum(
            hidden,
            self.W,
            "... instances hidden, instances hidden features -> ... instances features",
        )
        return F.relu(z + self.b_final)

    def forward(
        self, features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances features"]:
        hidden = einops.einsum(
            features,
            self.W,
            "... instances features, instances hidden features -> ... instances hidden",
        )
        out = einops.einsum(
            hidden,
            self.W,
            "... instances hidden, instances hidden features -> ... instances features",
        )
        return F.relu(out + self.b_final)

    # TODO: generate_batch_with_groups
    def generate_batch_with_groups(
        self,
        batch_size: int,
    ) -> Float[Tensor, "batch_size instances features"]:
        """
        Generates a batch of data where one feature per group is always active.
        """
        groups = self.cfg.groups
        # Generate random one-hot vector
        all_data = []
        for i, group_size in enumerate(groups):
            group_data: Float[Tensor, "batch_size instances n_group"] = t.zeros(
                (batch_size, self.cfg.n_instances, group_size), device=self.W.device
            )
            index = t.randint(group_size, (batch_size, self.cfg.n_instances))

            batch_size, instances, _ = group_data.shape
            for j in range(group_size):
                group_data[:, :, j] = (index == j) * torch.rand((batch_size, instances))

            # active_view = eindex(
            #     group_data, index, "batch_size instances [batch_size instances]"
            # )
            # batch_size, instances, _ = group_data.shape
            # active_view[:] = torch.rand(
            #     (
            #         batch_size,
            #         instances,
            #     )
            # )
            all_data.append(group_data)
        return t.cat(all_data, dim=-1)

    def generate_batch(
        self, batch_size
    ) -> Float[Tensor, "batch_size instances features"]:
        """
        Generates a batch of data where all features are randomly and independently active with some prob.
        We'll return to this function later when we apply correlations.
        """
        # Generate the features, before randomly setting some to zero
        feat = t.rand(
            (batch_size, self.cfg.n_instances, self.cfg.n_features),
            device=self.W.device,
        )

        # Generate a random boolean array, which is 1 wherever we'll keep a feature, and zero where we'll set it to zero
        feat_seeds = t.rand(
            (batch_size, self.cfg.n_instances, self.cfg.n_features),
            device=self.W.device,
        )
        feat_is_present = feat_seeds <= self.feature_probability

        # Create our batch from the features, where we set some to zero
        batch = t.where(feat_is_present, feat, 0.0)

        return batch

    def calculate_loss(
        self,
        out: Float[Tensor, "batch instances features"],
        batch: Float[Tensor, "batch instances features"],
    ) -> Float[Tensor, ""]:
        """
        Calculates the loss for a given batch, using this loss described in the Toy Models paper:

            https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss

        Remember, `self.importance` will always have shape (n_instances, n_features).
        """
        error = self.importance * ((batch - out) ** 2)
        loss = einops.reduce(
            error, "batch instances features -> instances", "mean"
        ).sum()
        return loss

    def optimize(
        self,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
    ):
        """
        Optimizes the model using the given hyperparameters.
        """
        optimizer = t.optim.Adam(list(self.parameters()), lr=lr)

        progress_bar = tqdm(range(steps))

        for step in progress_bar:
            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group["lr"] = step_lr

            # Optimize
            optimizer.zero_grad()
            batch = self.generate_batch(batch_size)
            out = self(batch)
            loss = self.calculate_loss(out, batch)
            loss.backward()
            optimizer.step()

            # Display progress bar
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(
                    loss=loss.item() / self.cfg.n_instances, lr=step_lr
                )
