"""Dropout-enabled copy of the AirfRANS flow model for MC dropout UQ.

Do not edit src/models.py — this file is the sanctioned copy.

Two changes vs MultiScaleModulatedFourierFeatures in src/models.py:

1. LatentToModulationDropout: the hypernetwork gets nn.Dropout between
   its hidden Linear+SiLU blocks (probability = dropout_hnn).

2. _shift_modulation_dropout: local copy of src/conditioning.py
   shift_modulation that applies nn.Dropout to the hidden activation h
   after each modulated layer (probability = dropout_trunk).

Both dropouts respect module .train() / .eval() automatically. MC
dropout usage: call model.train() before T stochastic forward passes;
call model.eval() for the deterministic point estimate.
"""
import torch
import torch.nn as nn

from src.models import GaussianEncoding


class LatentToModulationDropout(nn.Module):
    """Hypernetwork mapping latent -> modulations, with dropout between
    hidden Linear+SiLU blocks. num_layers=1 means a single Linear and
    no dropout (nothing to drop between)."""

    def __init__(self, latent_dim, num_modulations, dim_hidden, num_layers,
                 dropout_p=0.1, activation=nn.SiLU):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modulations = num_modulations
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        if num_layers == 1:
            self.net = nn.Linear(latent_dim, num_modulations)
        else:
            layers = [nn.Linear(latent_dim, dim_hidden), activation(),
                      nn.Dropout(dropout_p)]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(dim_hidden, dim_hidden), activation(),
                           nn.Dropout(dropout_p)]
            layers += [nn.Linear(dim_hidden, num_modulations)]
            self.net = nn.Sequential(*layers)

    def forward(self, latent):
        return self.net(latent)


def _shift_modulation_dropout(position, features, layers, activation,
                              dropout, with_batch=True):
    """Copy of src/conditioning.py:shift_modulation with dropout applied
    to the hidden activation after each modulated layer. `dropout` is an
    nn.Dropout module (respects .train()/.eval())."""
    feature_shape = features.shape[0]
    feature_dim = features.shape[-1]
    num_hidden = len(layers)

    if with_batch:
        features = features.reshape(feature_shape, 1, num_hidden,
                                    feature_dim // num_hidden)
    else:
        features = features.reshape(feature_shape, num_hidden,
                                    feature_dim // num_hidden)

    h = position
    for i, l in enumerate(layers):
        res = l(h)
        h = res * features[..., i, :] + features[..., i, :] + res
        h = activation(h)
        h = dropout(h)
    return h


class MultiScaleModulatedFourierFeaturesUQ(nn.Module):
    """Dropout-enabled copy of MultiScaleModulatedFourierFeatures.

    Constructor signature mirrors the original plus dropout_hnn and
    dropout_trunk. conditioning_type is fixed to 'shift_modulation'
    (the only path exercised by the AirfRANS pipeline)."""

    def __init__(
        self,
        input_dim=6,
        output_dim=4,
        num_frequencies=64,
        latent_dim=10,
        width=256,
        depth=6,
        depth_hnn=4,
        include_input=True,
        scales=(0.5, 1.0),
        scalar_out_dim=0,
        dropout_hnn=0.1,
        dropout_trunk=0.1,
    ):
        super().__init__()
        self.include_input = include_input
        self.scales = list(scales)

        self.embeddings = nn.ModuleList([
            GaussianEncoding(embedding_size=num_frequencies * 2, scale=s,
                             dims=input_dim)
            for s in self.scales
        ])
        embed_dim = num_frequencies * 2 + (input_dim if include_input else 0)
        self.in_channels = [embed_dim] + [width] * (depth - 1)
        self.out_channels = [width] * (depth - 1) + [width]

        self.latent_dim = latent_dim
        self.layers = nn.ModuleList([
            nn.Linear(self.in_channels[k], self.out_channels[k])
            for k in range(depth)
        ])
        self.final_linear = nn.Linear(len(self.scales) * width, output_dim)
        self.depth = depth
        self.hidden_dim = width
        self.depth_hnn = depth_hnn
        self.num_modulations = self.hidden_dim * (self.depth - 1)

        self.cond_to_modulation = LatentToModulationDropout(
            self.latent_dim, self.num_modulations,
            dim_hidden=256, num_layers=self.depth_hnn,
            dropout_p=dropout_hnn,
        )
        self.trunk_dropout = nn.Dropout(dropout_trunk)

        if scalar_out_dim is not None and scalar_out_dim > 0:
            self.scalar_head = nn.Sequential(
                nn.Linear(width * (depth - 1), 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, scalar_out_dim),
            )

    def modulated_forward(self, x, z):
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])

        features = self.cond_to_modulation(z)
        positions = [emb(x) for emb in self.embeddings]
        if self.include_input:
            positions = [torch.cat([p, x], axis=-1) for p in positions]

        pre_outs = [
            _shift_modulation_dropout(pos, features, self.layers[:-1],
                                      torch.relu, self.trunk_dropout)
            for pos in positions
        ]
        outs = [self.layers[-1](p) for p in pre_outs]
        concatenated = torch.cat(outs, axis=-1)
        final_out = self.final_linear(concatenated)
        return final_out.view(*x_shape, final_out.shape[-1])

    def predict_scalars(self, z):
        mod_feats = self.cond_to_modulation(z)
        return self.scalar_head(mod_feats)


def mc_predict(model, x, z, T=50):
    """T stochastic forward passes with dropout kept active. Returns
    (mean, std) each with the same shape as one forward output."""
    was_training = model.training
    model.train()
    with torch.no_grad():
        stack = torch.stack([model.modulated_forward(x, z) for _ in range(T)])
    if not was_training:
        model.eval()
    return stack.mean(dim=0), stack.std(dim=0)
