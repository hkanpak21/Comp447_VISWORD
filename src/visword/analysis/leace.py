"""Closed-form Least-squares Concept Erasure (LEACE) — Belrose et al.\
NeurIPS 2023.

Given features ``X`` and a protected attribute ``Z`` (one-hot or
continuous), LEACE returns the unique \emph{optimal} linear eraser
``M`` such that

* the linear probe for ``Z`` on ``X M`` recovers no signal beyond
  the attribute's mean (``Z`` becomes linearly unrecoverable);
* among all linear erasers with that property, ``M`` minimises
  ``E\\|X - X M\\|^2`` (least change to the features).

Implementation follows Eq.\ (8) of the LEACE paper. We do not use the
``concept-erasure`` PyPI package because it requires Python >= 3.10
and our cluster Python is 3.9.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


def _matrix_pow(S: torch.Tensor, p: float, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric matrix power S^p via eigendecomposition (S must be PSD)."""
    S = (S + S.T) * 0.5
    vals, vecs = torch.linalg.eigh(S)
    vals = torch.clamp(vals, min=eps)
    return vecs @ torch.diag(vals.pow(p)) @ vecs.T


@dataclass
class LeaceEraser:
    """The fitted eraser.  Apply with ``erase(X)``."""

    proj: torch.Tensor  # (d, d)
    mu_x: torch.Tensor  # (d,)

    def erase(self, X: torch.Tensor) -> torch.Tensor:
        """Apply the eraser to a batch of features."""
        return (X - self.mu_x) @ self.proj + self.mu_x


def fit_leace(X: torch.Tensor, Z: torch.Tensor,
              eps: float = 1e-6) -> LeaceEraser:
    """Fit a LEACE eraser.

    Args:
        X: ``(N, d)`` features (float32/float64).
        Z: ``(N, k)`` protected attribute (one-hot for categorical, raw
            for continuous; will be centered).
        eps: ridge added to the X covariance for stability.

    Returns:
        ``LeaceEraser`` such that the linear probe for ``Z`` on
        ``erase(X)`` recovers no signal beyond the attribute mean.
    """
    X = X.double()
    Z = Z.double()

    N, d = X.shape

    # Centering
    mu_x = X.mean(dim=0)
    mu_z = Z.mean(dim=0)
    Xc = X - mu_x
    Zc = Z - mu_z

    # X covariance + ridge
    SXX = (Xc.T @ Xc) / N + eps * torch.eye(d, dtype=X.dtype)

    # Whiten X via S^{-1/2}
    SXX_inv_half = _matrix_pow(SXX, -0.5, eps=eps)
    SXX_half = _matrix_pow(SXX, +0.5, eps=eps)

    # Cross-covariance in whitened space, (d, k)
    Cxz = SXX_inv_half @ (Xc.T @ Zc) / N

    # SVD of cross-cov: directions in whitened space correlated with Z
    U, S, _ = torch.linalg.svd(Cxz, full_matrices=False)

    # Keep components with non-negligible singular value
    rank = int((S > 1e-8).sum().item())
    Ur = U[:, :rank]                                   # (d, r)

    # Projection in whitened space: I - Ur Ur^T
    P_white = torch.eye(d, dtype=X.dtype) - Ur @ Ur.T

    # Compose the full eraser back in the original feature space:
    #   X_erased = mu + (X - mu) @ M, with M = SXX^{-1/2} P_white SXX^{1/2}.
    M = SXX_inv_half @ P_white @ SXX_half

    return LeaceEraser(proj=M.float(), mu_x=mu_x.float())


# ---------------------------------------------------------------------------
# Sanity test (run as `python -m visword.analysis.leace` for a smoke check)
# ---------------------------------------------------------------------------


def _smoke_test():
    """Construct synthetic data where Z is a 2-class label perfectly
    encoded as the sign of the first feature dimension, then verify
    that after LEACE the linear probe is at chance."""
    torch.manual_seed(0)
    N, d = 2000, 32
    Z = torch.randint(0, 2, (N,))
    X = torch.randn(N, d)
    X[:, 0] = (Z.float() - 0.5) * 4.0 + 0.1 * X[:, 0]  # encode Z in dim 0
    Z_oh = torch.nn.functional.one_hot(Z, 2).float()

    eraser = fit_leace(X, Z_oh)
    X_e = eraser.erase(X)

    # Linear probe for Z BEFORE erasure (logistic regression closed-form
    # via least squares on one-hot is fine for a sanity check).
    def acc(X_):
        coef = torch.linalg.lstsq(X_, Z_oh).solution    # (d, 2)
        pred = (X_ @ coef).argmax(dim=1)
        return float((pred == Z).float().mean())

    a_before = acc(X)
    a_after  = acc(X_e)
    print(f"  probe accuracy BEFORE erasure: {a_before:.3f}  (ideal: ~1.0)")
    print(f"  probe accuracy AFTER  erasure: {a_after:.3f}  (ideal: ~0.5 chance)")
    # Verify minimum change
    mse = float(((X - X_e) ** 2).mean())
    print(f"  mean-squared change: {mse:.4f}")
    assert a_before > 0.95, "BEFORE accuracy should be near 1.0"
    assert a_after < 0.6, "AFTER accuracy should be near chance 0.5"
    print("OK: LEACE erases the protected attribute as expected.")


if __name__ == "__main__":
    _smoke_test()
