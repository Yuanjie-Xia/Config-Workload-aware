from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def mre(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """Mean Relative Error in %, common in performance modeling papers."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs(y_pred - y_true) / np.clip(np.abs(y_true), eps, None)) * 100.0)


@dataclass
class DeepPerfParams:
    n_layers: int = 3
    hidden_units: int = 64
    activation: str = "relu"      # "relu", "tanh", "gelu"
    l1: float = 1e-4              # L1 coeff for sparsity
    dropout: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 200
    patience: int = 20
    val_size: float = 0.2
    seed: int = 42
    normalize_y: bool = False     # scale targets with StandardScaler


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_layers: int, hidden: int,
                 activation: str, dropout: float):
        super().__init__()
        act: Dict[str, nn.Module] = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
        }
        a = act.get(activation, nn.ReLU())
        layers: List[nn.Module] = []
        dim_in = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(dim_in, hidden), a]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            dim_in = hidden
        layers += [nn.Linear(dim_in, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepPerfRegressor(BaseEstimator, RegressorMixin):
    """
    PyTorch sparse MLP regressor inspired by DeepPerf (ICSE'19):
      - L1 regularization on weights (feature sparsity effect)
      - Early stopping on val loss
      - Optional random search tuning (tune=N)

    API:
      reg = DeepPerfRegressor(n_layers=4, hidden_units=128, l1=1e-3, random_state=42)
      reg.fit(X_train, y_train, tune=30)
      y_pred = reg.predict(X_test)
      reg.score(X_test, y_test)         # R^2
      reg.mre_score(X_test, y_test)     # % MRE
    """
    def __init__(
        self,
        n_layers: int = 3,
        hidden_units: int = 64,
        activation: str = "relu",
        l1: float = 1e-4,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        epochs: int = 200,
        patience: int = 20,
        val_size: float = 0.2,
        normalize_y: bool = False,
        random_state: Optional[int] = 42,
        device: Optional[str] = None,    # "cuda", "cpu", or None for auto
    ):
        self.params = DeepPerfParams(
            n_layers=n_layers, hidden_units=hidden_units, activation=activation,
            l1=l1, dropout=dropout, learning_rate=learning_rate,
            batch_size=batch_size, epochs=epochs, patience=patience,
            val_size=val_size, seed=(random_state if random_state is not None else 42),
            normalize_y=normalize_y
        )
        self.random_state = random_state
        self.device = device
        self._scaler_X: Optional[StandardScaler] = None
        self._scaler_y: Optional[StandardScaler] = None
        self._model: Optional[_MLP] = None
        self._in_dim: Optional[int] = None

    # sklearn plumbing
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        d = self.params.__dict__.copy()
        d.pop("seed", None)
        d["random_state"] = self.random_state
        d["device"] = self.device
        return d

    def set_params(self, **params):
        if "seed" in params and "random_state" not in params:
            params["random_state"] = params.pop("seed")

        for k, v in params.items():
            if hasattr(self.params, k):
                setattr(self.params, k, v)
            elif k in ("random_state", "device"):
                if k == "random_state":
                    self.random_state = v
                    self.params.seed = v
                else:
                    self.device = v
            else:
                raise ValueError(f"Unknown param: {k}")
        return self

    # core
    def _choose_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, n_features: int) -> _MLP:
        torch.manual_seed(self.params.seed)
        model = _MLP(
            in_dim=n_features, out_dim=1,
            n_layers=self.params.n_layers,
            hidden=self.params.hidden_units,
            activation=self.params.activation,
            dropout=self.params.dropout
        )
        return model

    def _l1_penalty(self, model: nn.Module) -> torch.Tensor:
        l1 = torch.tensor(0.0, device=next(model.parameters()).device)
        if self.params.l1 and self.params.l1 > 0:
            for name, p in model.named_parameters():
                if "weight" in name and p.requires_grad:
                    l1 = l1 + p.abs().sum()
            l1 = self.params.l1 * l1
        return l1

    def _inv_y(self, y_scaled: np.ndarray) -> np.ndarray:
        if self._scaler_y is not None:
            return self._scaler_y.inverse_transform(y_scaled)
        return y_scaled

    def _train_once(self, Xtr, ytr, Xval, yval, verbose: int) -> Tuple[_MLP, float]:
        dev = self._choose_device()
        model = self._build_model(Xtr.shape[1]).to(dev)

        # data
        tr_ds = TensorDataset(
            torch.from_numpy(Xtr.astype(np.float32)),
            torch.from_numpy(ytr.astype(np.float32)).view(-1, 1)
        )
        val_ds = TensorDataset(
            torch.from_numpy(Xval.astype(np.float32)),
            torch.from_numpy(yval.astype(np.float32)).view(-1, 1)
        )
        tr_loader = DataLoader(tr_ds, batch_size=self.params.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.params.batch_size, shuffle=False)

        opt = optim.Adam(model.parameters(), lr=self.params.learning_rate)
        mse = nn.MSELoss()

        best_loss = math.inf
        best_state = None
        wait = 0

        for epoch in range(self.params.epochs):
            model.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                pred = model(xb)
                loss = mse(pred, yb) + self._l1_penalty(model)
                loss.backward()
                opt.step()

            # validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(dev), yb.to(dev)
                    pred = model(xb)
                    vloss = mse(pred, yb) + self._l1_penalty(model)  # same regularization
                    val_losses.append(vloss.item())
            mean_vloss = float(np.mean(val_losses))

            if mean_vloss < best_loss - 1e-8:
                best_loss = mean_vloss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.params.patience:
                    break

            if verbose and (epoch % max(1, self.params.epochs // 10) == 0):
                print(f"[epoch {epoch:04d}] val_loss={mean_vloss:.6f}")

        # restore best
        if best_state is not None:
            model.load_state_dict(best_state)

        # choose by validation MRE (like DeepPerf reporting)
        with torch.no_grad():
            Xv = torch.from_numpy(Xval.astype(np.float32)).to(dev)
            yhat_v = model(Xv).cpu().numpy()
        yhat_v = self._inv_y(yhat_v)
        m = mre(self._inv_y(yval), yhat_v)
        return model, m

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tune: Optional[int] = None,
        param_space: Optional[Dict[str, Iterable]] = None,
        verbose: int = 0
    ):
        rng = np.random.default_rng(self.params.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        # scale features (and optionally targets)
        self._scaler_X = StandardScaler()
        Xs = self._scaler_X.fit_transform(X)

        ys = y
        if self.params.normalize_y:
            self._scaler_y = StandardScaler()
            ys = self._scaler_y.fit_transform(y)

        Xtr, Xval, ytr, yval = train_test_split(
            Xs, ys, test_size=self.params.val_size, random_state=self.params.seed
        )

        def train_with_current_params():
            return self._train_once(Xtr, ytr, Xval, yval, verbose)

        # random search tuner
        if tune and tune > 0:
            if param_space is None:
                param_space = {
                    "n_layers": [2, 3, 4, 5],
                    "hidden_units": [32, 64, 128, 256],
                    "l1": [1e-5, 1e-4, 1e-3, 1e-2],
                    "learning_rate": [3e-4, 1e-3, 3e-3],
                    "dropout": [0.0, 0.1, 0.2, 0.3],
                    "activation": ["relu", "tanh", "gelu"],
                }

            best_mre = math.inf
            best_state = None
            best_params = self.params

            for _ in range(tune):
                trial = {k: rng.choice(list(v)) for k, v in param_space.items()}
                self.set_params(**trial)
                model, val_mre = train_with_current_params()
                if val_mre < best_mre:
                    best_mre = val_mre
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    best_params = DeepPerfParams(**self.params.__dict__)

            # finalize with best
            self.params = best_params
            self._model = self._build_model(Xs.shape[1])
            self._model.load_state_dict(best_state)
        else:
            self._model, _ = train_with_current_params()

        self._in_dim = Xs.shape[1]
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Call fit() first."
        X = np.asarray(X, dtype=np.float32)
        Xs = self._scaler_X.transform(X)
        dev = self._choose_device()
        self._model.to(dev).eval()
        yhat = self._model(torch.from_numpy(Xs).to(dev)).cpu().numpy().reshape(-1, 1)
        return self._inv_y(yhat).reshape(-1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y).reshape(-1)
        yhat = self.predict(X)
        return r2_score(y, yhat)

    def mre_score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y).reshape(-1)
        yhat = self.predict(X)
        return mre(y, yhat)

    # convenience
    def save(self, path: str):
        """Save scalers + model state."""
        payload = {
            "state_dict": self._model.state_dict() if self._model else None,
            "in_dim": self._in_dim,
            "params": self.params.__dict__,
            "random_state": self.random_state,
            "device": self.device,
            "scaler_X_mean": self._scaler_X.mean_.tolist() if self._scaler_X else None,
            "scaler_X_scale": self._scaler_X.scale_.tolist() if self._scaler_X else None,
            "scaler_y_mean": (self._scaler_y.mean_.tolist() if self._scaler_y else None),
            "scaler_y_scale": (self._scaler_y.scale_.tolist() if self._scaler_y else None),
        }
        torch.save(payload, path)

    def load(self, path: str):
        """Load scalers + model state into this instance."""
        payload = torch.load(path, map_location="cpu")
        self.params = DeepPerfParams(**payload["params"])
        self.random_state = payload["random_state"]
        self.device = payload["device"]
        # rebuild model
        self._in_dim = payload["in_dim"]
        self._model = self._build_model(self._in_dim)
        if payload["state_dict"]:
            self._model.load_state_dict(payload["state_dict"])
        # restore scalers
        self._scaler_X = StandardScaler()
        self._scaler_X.mean_ = np.array(payload["scaler_X_mean"], dtype=np.float64)
        self._scaler_X.scale_ = np.array(payload["scaler_X_scale"], dtype=np.float64)
        self._scaler_X.var_ = self._scaler_X.scale_ ** 2
        if payload["scaler_y_mean"] is not None:
            self._scaler_y = StandardScaler()
            self._scaler_y.mean_ = np.array(payload["scaler_y_mean"], dtype=np.float64)
            self._scaler_y.scale_ = np.array(payload["scaler_y_scale"], dtype=np.float64)
            self._scaler_y.var_ = self._scaler_y.scale_ ** 2
        return self
