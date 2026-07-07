"""
Neural network models for the Cocycle PINN framework.

FNO1d:           Fourier Neural Operator — equation-agnostic.
SemigroupFNO1d:  Time-conditioned semigroup operator Φ_θ(state, Δ).
                 Skip-connection form:  Φ(s, Δ) = s + Δ * G(s, Δ)
                 where G is an FNO1d with (state_channels + 1) inputs.

Both are parameterized by (modes, width, n_layers, state_channels) so the
same code works for wave (state_channels=2), KS (state_channels=1), etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ───────────────────────────────────────────────────────────

class SpectralConv1d(nn.Module):
    """FFT → truncate to modes → complex multiply → IFFT."""

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes        = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat) * scale
        )

    def forward(self, x):
        """x: (B, N, in_channels) → (B, N, out_channels)"""
        B, N, _ = x.shape
        x_ft    = torch.fft.fft(x, dim=1)
        weights = self.weights.to(dtype=x_ft.dtype)

        out_ft = torch.zeros(B, N, self.out_channels,
                             dtype=x_ft.dtype, device=x.device)
        out_ft[:, :self.modes, :] = torch.einsum(
            'bmi,iom->bmo', x_ft[:, :self.modes, :], weights
        )
        return torch.fft.ifft(out_ft, dim=1).real


class FourierLayer(nn.Module):
    """SpectralConv + pointwise Linear + GELU."""

    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.linear   = nn.Linear(width, width)

    def forward(self, x, apply_activation=True):
        """x: (B, N, width)"""
        h = self.spectral(x) + self.linear(x)
        if apply_activation:
            h = F.gelu(h)
        return h


# ── FNO1d ─────────────────────────────────────────────────────────────────────

class FNO1d(nn.Module):
    """
    Fourier Neural Operator for 1D problems.

    Input:  (B, N, in_channels)
    Output: (B, N, out_channels)
    """

    def __init__(self, modes=16, width=64, n_layers=4,
                 in_channels=1, out_channels=1):
        super().__init__()
        self.modes       = modes
        self.width       = width
        self.n_layers    = n_layers
        self.in_channels  = in_channels
        self.out_channels = out_channels

        self.lift = nn.Linear(in_channels, width)
        self.fourier_layers = nn.ModuleList([
            FourierLayer(width, modes) for _ in range(n_layers)
        ])
        self.proj1 = nn.Linear(width, 128)
        self.proj2 = nn.Linear(128, out_channels)

    def forward(self, x):
        """x: (B, N, in_channels) → (B, N, out_channels)"""
        h = self.lift(x)
        for i, layer in enumerate(self.fourier_layers):
            h = layer(h, apply_activation=(i < self.n_layers - 1))
        return self.proj2(F.gelu(self.proj1(h)))


# ── SemigroupFNO1d ────────────────────────────────────────────────────────────

class SemigroupFNO1d(nn.Module):
    """
    Time-conditioned semigroup operator  Φ_θ(state, Δ).

    Skip-connection form:
        Φ_θ(s, Δ) = s + Δ * G_θ(s, Δ)

    G_θ is an FNO1d with (state_channels + 1) inputs (field + Δt) and
    state_channels outputs (δstate/step).

    Args:
        modes:          Fourier modes to retain
        width:          channel width inside FNO
        n_layers:       number of Fourier layers
        state_channels: dimensionality of state (2 for wave, 1 for KS)
    """

    def __init__(self, modes=16, width=64, n_layers=4, state_channels=2,
                 forcing_channels=0):
        super().__init__()
        self.state_channels   = state_channels
        self.forcing_channels = int(forcing_channels)
        self.core = FNO1d(
            modes=modes, width=width, n_layers=n_layers,
            # state + Δt + (optional) forcing channels
            in_channels=state_channels + 1 + self.forcing_channels,
            out_channels=state_channels,
        )

    def _broadcast_delta(self, delta_steps, B, device, dtype):
        if delta_steps is None:
            return torch.ones(B, device=device, dtype=dtype)
        if isinstance(delta_steps, (int, float)):
            return torch.full((B,), float(delta_steps), device=device, dtype=dtype)
        delta_steps = torch.as_tensor(delta_steps, device=device, dtype=dtype)
        if delta_steps.dim() == 0:
            return delta_steps.repeat(B)
        if delta_steps.dim() == 1 and delta_steps.shape[0] == B:
            return delta_steps
        raise ValueError(
            f"delta_steps must be scalar or (B,), got {tuple(delta_steps.shape)}"
        )

    def forward(self, state, delta_steps=None, forcing=None):
        """
        Args:
            state:       (B, N, state_channels)
            delta_steps: scalar or (B,) — number of base time steps
            forcing:     (B, N, forcing_channels) — required iff forcing_channels > 0

        Returns:
            (B, N, state_channels)  next state
        """
        if state.dim() != 3 or state.shape[-1] != self.state_channels:
            raise ValueError(
                f"Expected state (B, N, {self.state_channels}), got {tuple(state.shape)}"
            )
        B, N, _ = state.shape
        steps   = self._broadcast_delta(delta_steps, B, state.device, state.dtype)

        step_feat = steps.view(B, 1, 1).expand(B, N, 1)
        feats     = [state, step_feat]

        if self.forcing_channels > 0:
            if forcing is None:
                raise ValueError(
                    f"Model expects forcing (forcing_channels={self.forcing_channels}) "
                    f"but none was supplied."
                )
            if forcing.dim() == 2:           # (B, N) → (B, N, 1)
                forcing = forcing.unsqueeze(-1)
            if forcing.shape[0] != B or forcing.shape[1] != N \
               or forcing.shape[2] != self.forcing_channels:
                raise ValueError(
                    f"forcing shape {tuple(forcing.shape)} does not match "
                    f"(B={B}, N={N}, forcing_channels={self.forcing_channels})"
                )
            feats.append(forcing.to(dtype=state.dtype, device=state.device))
        elif forcing is not None:
            # Silently ignore if the model wasn't configured for forcing — keeps
            # the same call site usable across forcing-aware and forcing-free models.
            pass

        inp = torch.cat(feats, dim=-1)                       # (B, N, in_channels)
        delta_per_step = self.core(inp)                      # (B, N, state_channels)
        return state + steps.view(B, 1, 1) * delta_per_step
    
    def load_checkpoint(self, checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])

        meta_data = {
            'all_train_losses': checkpoint['train_losses'],
            'all_val_losses': checkpoint['val_losses'],
            'iteration': checkpoint['iteration'],
            'best_iteration': checkpoint['best_iteration'],
            'best_val_loss': checkpoint['best_val_loss']
        }

        return meta_data


# ── Weight transfer ───────────────────────────────────────────────────────────

def init_semigroup_from_fno1d(semigroup_model: SemigroupFNO1d,
                               fno1d_path: str,
                               device=None) -> None:
    """
    Warm-start a SemigroupFNO1d from a pre-trained plain FNO1d checkpoint.

    The plain FNO1d has in_channels=state_channels; the semigroup has
    in_channels=state_channels+1. The extra Δt column of the lift weight
    is zero-initialised so the model initially ignores time conditioning
    and learns it gradually.
    """
    if device is None:
        device = next(semigroup_model.parameters()).device

    payload = torch.load(fno1d_path, map_location=device)
    src = (payload['model_state']
           if isinstance(payload, dict) and 'model_state' in payload
           else payload)

    dst = semigroup_model.state_dict()
    transferred, skipped = [], []

    for src_key, src_val in src.items():
        dst_key = f'core.{src_key}'
        if dst_key not in dst:
            skipped.append(dst_key)
            continue

        dst_val = dst[dst_key]
        if dst_val.shape == src_val.shape:
            dst[dst_key] = src_val.to(dst_val.dtype)
            transferred.append(dst_key)
        elif dst_key == 'core.lift.weight':
            # src: (width, state_channels) → dst: (width, state_channels+1)
            new_w = torch.zeros_like(dst_val)
            new_w[:, :src_val.shape[1]] = src_val.to(dst_val.dtype)
            dst[dst_key] = new_w
            transferred.append(dst_key)
        else:
            skipped.append(f'{dst_key} (shape mismatch)')

    semigroup_model.load_state_dict(dst)
    print(f"  Transferred {len(transferred)} tensors from plain FNO1d.")
    if skipped:
        print(f"  Skipped: {skipped}")


def build_model(cfg, device='cpu'):
    m = cfg['model']
    return SemigroupFNO1d(
        modes=int(m['modes']), width=int(m['width']),
        n_layers=int(m['n_layers']), state_channels=int(m['state_channels']),
        forcing_channels=int(m.get('forcing_channels', 0)),
    ).double().to(device)


