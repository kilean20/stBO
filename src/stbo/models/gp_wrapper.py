from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional

import torch
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms import Normalize, Standardize
from botorch.optim.utils import get_parameters_and_bounds
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from .priors import PriorMean


class BoTorchGPWrapper(Model):
    """A thin wrapper providing:
    - Optional additive prior mean model
    - Choice of kernel
    - History for torch-based optimizer runs
    - A `posterior` method that returns residual+prior as a GPyTorchPosterior
    """

    def __init__(self, kernel_type: str = "matern", prior_mean: Optional[PriorMean] = None):
        super().__init__()
        self.kernel_type = kernel_type
        self.prior_mean = prior_mean

        self.model: Optional[SingleTaskGP] = None
        self.mll: Optional[ExactMarginalLogLikelihood] = None
        self.train_history = {"fit_gpytorch_mll_torch": []}

    @property
    def num_outputs(self) -> int:
        return self.model.num_outputs if self.model else 1

    @property
    def batch_shape(self) -> torch.Size:
        return self.model.batch_shape if self.model else torch.Size()

    def clear_history(self, method: str = "fit_gpytorch_mll_torch") -> None:
        self.train_history.setdefault(method, [])
        self.train_history[method].clear()

    def get_history(self, method: str = "fit_gpytorch_mll_torch"):
        return list(self.train_history.get(method, []))

    def _fit_with_fit_gpytorch_mll_torch(
        self,
        *,
        step_limit: int = 200,
        lr: float = 0.05,
        betas=(0.9, 0.999),
        weight_decay: float = 0.0,
        reset_history: bool = True,
        preclamp_to_bounds: bool = True,
    ) -> None:
        if self.model is None or self.mll is None:
            raise RuntimeError("Model/MLL not initialized. Call fit(...) first.")

        self.model.train()
        self.model.likelihood.train()
        self.mll.train()

        method_key = "fit_gpytorch_mll_torch"
        if reset_history:
            self.clear_history(method_key)

        if preclamp_to_bounds:
            param_dict, bounds_dict = get_parameters_and_bounds(self.mll)
            for name, p in param_dict.items():
                if not p.requires_grad:
                    continue
                if name not in bounds_dict:
                    continue
                lb, ub = bounds_dict[name]
                lo = -float("inf") if lb is None else float(lb)
                hi = float("inf") if ub is None else float(ub)
                with torch.no_grad():
                    p.data.clamp_(min=lo, max=hi)

        def cb(_params: Dict[str, torch.Tensor], result) -> None:
            step = int(result.step)
            self.train_history[method_key].append({"step": step, "loss": float(result.fval)})

        opt_factory = partial(torch.optim.Adam, lr=lr, betas=betas, weight_decay=weight_decay)

        fit_gpytorch_mll_torch(
            self.mll,
            step_limit=int(step_limit),
            optimizer=opt_factory,
            callback=cb,
        )

        self.model.eval()
        self.model.likelihood.eval()
        self.mll.eval()

    def fit(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        bounds: torch.Tensor,
        *,
        fresh_train: bool = True,
        train_method: str = "fit_gpytorch_mll_torch",
        torch_lr: float = 0.01,
        torch_steps: int = 250,
    ) -> None:
        targets = train_y
        if self.prior_mean is not None:
            self.prior_mean.eval()
            with torch.no_grad():
                prior_pred = self.prior_mean(train_x)
                targets = train_y - prior_pred.view_as(train_y)

        input_dim = train_x.shape[-1]
        if self.kernel_type.lower() == "matern":
            cov_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=input_dim))
        else:
            cov_module = ScaleKernel(RBFKernel(ard_num_dims=input_dim))

        new_model = SingleTaskGP(
            train_x,
            targets,
            covar_module=cov_module,
            input_transform=Normalize(d=input_dim, bounds=bounds),
            outcome_transform=Standardize(m=1),
        )

        # Warm start: ignore transform stats
        if (not fresh_train) and (self.model is not None):
            old_state = self.model.state_dict()
            filtered_state = {k: v for k, v in old_state.items() if "transform" not in k}
            new_model.load_state_dict(filtered_state, strict=False)

        self.model = new_model
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        method = (train_method or "").lower()
        if method in ("fit_gpytorch_mll", "botorch", "lbfgs"):
            fit_gpytorch_mll(self.mll)
            self.model.eval()
            self.model.likelihood.eval()
            self.mll.eval()
        elif method in ("fit_gpytorch_mll_torch", "torch"):
            self._fit_with_fit_gpytorch_mll_torch(
                step_limit=torch_steps,
                lr=torch_lr,
                reset_history=True,
                preclamp_to_bounds=True,
            )
        else:
            raise ValueError(f"Unknown train_method={train_method!r}.")

    def posterior(
        self,
        X: torch.Tensor,
        observation_noise: bool = False,
        posterior_transform=None,
    ) -> GPyTorchPosterior:
        """Return a posterior over the full function (residual + prior mean)."""
        if self.model is None:
            raise RuntimeError("Underlying GP not fit yet. Call fit() before posterior().")

        post_residual = self.model.posterior(
            X, observation_noise=observation_noise, posterior_transform=posterior_transform
        )

        if self.prior_mean is None:
            return post_residual

        mvn_residual = post_residual.distribution
        with torch.no_grad():
            prior_mu = self.prior_mean(X)

        shifted_mean = mvn_residual.mean + prior_mu.view_as(mvn_residual.mean)
        shifted_mvn = MultivariateNormal(shifted_mean, mvn_residual.lazy_covariance_matrix)
        return GPyTorchPosterior(shifted_mvn)
