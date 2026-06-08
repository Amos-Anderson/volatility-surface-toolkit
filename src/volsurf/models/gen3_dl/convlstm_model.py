"""Conv2D surface completion network (ConvLSTM-inspired).

A U-Net-style encoder--decoder that learns to complete a volatility
surface from sparse observed IV values.  Adapted for single-snapshot
data (spatial Conv only; temporal LSTM left for future multi-day work).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSurfaceNet(nn.Module):
    """Conv2D encoder--decoder for volatility surface completion.

    Parameters
    ----------
    grid_shape : tuple[int, int]
        (n_strikes, n_maturities) of the input/output grids.
    channels : list[int]
        Encoder channel progression (default [32, 64, 128]).
    bottleneck_dim : int
        Fully-connected bottleneck dimension (default 512).
    smooth_lambda : float
        Weight of the Laplacian smoothness penalty (default 1e-3).
    """

    def __init__(
        self,
        grid_shape=(40, 11),
        channels=(32, 64, 128),
        bottleneck_dim=512,
        smooth_lambda=1e-3,
    ):
        super().__init__()
        self.grid_shape = grid_shape
        self.smooth_lambda = smooth_lambda
        c_in = 2  # IV grid + mask grid

        # --- Encoder ---
        self.enc_blocks = nn.ModuleList()
        prev_c = c_in
        for c_out in channels:
            self.enc_blocks.append(
                nn.Sequential(
                    nn.Conv2d(prev_c, c_out, kernel_size=3, padding=1),
                    nn.GroupNorm(1, c_out),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
                    nn.GroupNorm(1, c_out),
                    nn.ReLU(inplace=True),
                )
            )
            prev_c = c_out

        # --- Bottleneck ---
        self.flat_dim = channels[-1] * grid_shape[0] * grid_shape[1]
        self.fc1 = nn.Linear(self.flat_dim, bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, self.flat_dim)

        # --- Decoder ---
        self.dec_blocks = nn.ModuleList()
        rev_channels = list(reversed(channels))
        for i, c_out in enumerate(rev_channels):
            c_in_dec = rev_channels[i] if i == 0 else rev_channels[i - 1]
            self.dec_blocks.append(
                nn.Sequential(
                    nn.Conv2d(c_in_dec, c_out, kernel_size=3, padding=1),
                    nn.GroupNorm(1, c_out),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
                    nn.GroupNorm(1, c_out),
                    nn.ReLU(inplace=True),
                )
            )

        # Final 1-channel output
        self.head = nn.Conv2d(rev_channels[-1], 1, kernel_size=3, padding=1)

        self._count_params()

    def _count_params(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"ConvSurfaceNet: {n:,} trainable parameters")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 2, n_K, n_T)
            Channel 0 = IV values (0 where missing).
            Channel 1 = binary mask (1 where observed).

        Returns
        -------
        torch.Tensor, shape (B, 1, n_K, n_T)
            Completed surface.
        """
        h = x
        # Encoder
        for block in self.enc_blocks:
            h = block(h)

        # Bottleneck
        b, c, hk, ht = h.shape
        h = h.view(b, -1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = h.view(b, c, hk, ht)

        # Decoder
        for block in self.dec_blocks:
            h = block(h)

        # Output head
        out = self.head(h)
        return out

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, pred, target, mask):
        """Masked MSE + Laplacian smoothness.

        Parameters
        ----------
        pred, target : torch.Tensor, shape (B, 1, n_K, n_T)
        mask : torch.Tensor, shape (B, 1, n_K, n_T), bool or float
            1 = observed (compute MSE), 0 = missing (ignore).

        Returns
        -------
        dict with keys 'total', 'data', 'smooth'
        """
        # Data fidelity: MSE on observed positions only
        diff = (pred - target) * mask
        n_obs = mask.sum() + 1e-8
        loss_data = (diff ** 2).sum() / n_obs

        # Laplacian smoothness (penalise second derivatives)
        lap_k = self._laplacian_1d(pred, dim=2)  # strike direction
        lap_t = self._laplacian_1d(pred, dim=3)  # maturity direction
        loss_smooth = (lap_k ** 2).mean() + (lap_t ** 2).mean()

        total = loss_data + self.smooth_lambda * loss_smooth
        return {
            "total": total,
            "data": loss_data,
            "smooth": loss_smooth,
        }

    @staticmethod
    def _laplacian_1d(x, dim):
        """Finite-difference Laplacian along dimension ``dim``."""
        # Second derivative: x[i-1] - 2x[i] + x[i+1]
        pad = (0, 0, 0, 0)
        if dim == 2:   # strike dimension (dim 2 of 4D tensor)
            pad = (0, 0, 1, 1)
        elif dim == 3:  # maturity dimension (dim 3)
            pad = (1, 1, 0, 0)
        x_padded = F.pad(x, pad, mode="replicate")
        if dim == 2:
            return x_padded[:, :, :-2, :] - 2 * x + x_padded[:, :, 2:, :]
        else:
            return x_padded[:, :, :, :-2] - 2 * x + x_padded[:, :, :, 2:]

    # ------------------------------------------------------------------
    # Grid helper
    # ------------------------------------------------------------------

    def prepare_grid(
        self,
        strikes,
        maturities,
        iv_values,
        n_strike_bins=None,
        n_mat_bins=None,
    ):
        """Map irregular (strike, maturity, IV) triples to a regular grid.

        Parameters
        ----------
        strikes, maturities, iv_values : array-like, same length
        n_strike_bins : int or None
            Defaults to self.grid_shape[0].
        n_mat_bins : int or None
            Defaults to self.grid_shape[1].

        Returns
        -------
        iv_grid : np.ndarray, shape (n_K, n_T)
        mask_grid : np.ndarray, shape (n_K, n_T)
        strike_centers : np.ndarray, shape (n_K,)
        mat_centers : np.ndarray, shape (n_T,)
        """
        n_k = n_strike_bins or self.grid_shape[0]
        n_t = n_mat_bins or self.grid_shape[1]

        k_bins = np.linspace(strikes.min(), strikes.max(), n_k + 1)
        t_bins = np.linspace(maturities.min(), maturities.max(), n_t + 1)

        k_idx = np.digitize(strikes, k_bins) - 1
        t_idx = np.digitize(maturities, t_bins) - 1
        k_idx = np.clip(k_idx, 0, n_k - 1)
        t_idx = np.clip(t_idx, 0, n_t - 1)

        iv_grid = np.zeros((n_k, n_t))
        mask_grid = np.zeros((n_k, n_t))

        for i in range(len(strikes)):
            ki, ti = k_idx[i], t_idx[i]
            if mask_grid[ki, ti] == 0:
                iv_grid[ki, ti] = iv_values[i]
                mask_grid[ki, ti] = 1
            else:
                # Average multiple quotes in same cell
                iv_grid[ki, ti] = (iv_grid[ki, ti] * mask_grid[ki, ti] + iv_values[i]) / (mask_grid[ki, ti] + 1)
                mask_grid[ki, ti] += 1

        # Normalise mask to 0/1
        mask_grid = (mask_grid > 0).astype(np.float32)

        k_centers = (k_bins[:-1] + k_bins[1:]) / 2
        t_centers = (t_bins[:-1] + t_bins[1:]) / 2

        return iv_grid, mask_grid, k_centers, t_centers

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_surface(self, iv_grid, mask_grid):
        """Complete a surface from a partial observation.

        Parameters
        ----------
        iv_grid, mask_grid : np.ndarray, shape (n_K, n_T)

        Returns
        -------
        np.ndarray, shape (n_K, n_T)
            Completed surface.
        """
        self.eval()
        x = np.stack([iv_grid, mask_grid], axis=0)  # (2, n_K, n_T)
        x = torch.from_numpy(x).unsqueeze(0).float()  # (1, 2, n_K, n_T)
        pred = self.forward(x)
        return pred.squeeze().cpu().numpy()
