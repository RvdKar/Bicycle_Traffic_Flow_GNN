from __future__ import annotations
import math
from typing import Dict
from tqdm.auto import tqdm
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
# from preprocessing import Config
from . import preprocessing, features

def spike_aware_loss(pred, target, mask, tau, alpha=1.5, beta=0.3):
    # base MAE, upweighted for large targets
    w = 1.0 + alpha * (target / tau).clamp(min=0)
    base = (pred - target).abs()
    num = (base * w * mask).sum()
    den = (w * mask).sum().clamp_min(1.0)
    loss_base = num / den
    # slope term (match first differences)
    d_pred = pred[:, 1:, :] - pred[:, :-1, :]
    d_tgt  = target[:, 1:, :] - target[:, :-1, :]
    d_mask = mask[:, 1:, :] * mask[:, :-1, :]
    loss_slope = (d_mask * (d_pred - d_tgt).abs()).sum() / d_mask.sum().clamp_min(1.0)
    return loss_base + beta * loss_slope


def masked_mae(pred, target, mask, eps=1e-8):
    # pred/target: [B,H,E]; mask: [B,H,E]
    diff = (pred - target).abs()
    num = (diff * mask).sum()
    den = mask.sum() + eps
    return num / den


def masked_mse(pred, target, mask, eps=1e-8):
    mask = (mask > 0).to(pred.dtype)
    diff2 = (pred - target) ** 2
    num = (diff2 * mask).sum()
    den = mask.sum().clamp_min(1.0)
    return num / den


def train_model(model, loaders, config: Config, epoch_hook=None, A_bin: np.ndarray | None = None):


    device = config.device
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=config.lr)
    best = {"val": float("inf"), "state": None, "epoch": -1}
    patience = config.patience

    hist = {"epoch": [], "train": [], "val": []}

    for epoch in tqdm(range(1, config.max_epochs + 1), desc="Epochs", leave=True):
        model.train()
        tot = 0.0

        for batch in loaders["train"]:
            x = batch["x"].to(device, non_blocking=True)              # [B,H_in,E,F]
            y = batch["y"].to(device, non_blocking=True)              # [B,H_out,E]
            m = batch["mask"].to(device, non_blocking=True)           # [B,H_out,E]
            
            # quick checks
            for name, t in [("x", x), ("y", y), ("mask", m)]:
                if not torch.isfinite(t).all():
                    bad = (~torch.isfinite(t)).sum().item()
                    raise RuntimeError(f"{name} has {bad} non-finite values")

            opt.zero_grad()
            yhat = model(x)
            if not torch.isfinite(yhat).all():
                raise RuntimeError("model output contains NaN/Inf")
            loss = masked_mae(yhat, y, m)
            if not torch.isfinite(loss):
                print("Non-finite loss. Stats:",
                    f"y min/max = {float(y.min()):.3f}/{float(y.max()):.3f},",
                    f"mask sum = {float(m.sum()):.1f}")
                raise RuntimeError("Loss is NaN/Inf")
            
            # Laplacian smoothing (optional)
            if getattr(config, "lambda_lap", 0.0) > 0.0:
                if A_bin is None:
                    # optional: avoid silent failure
                    raise ValueError("lambda_lap > 0 but A_bin=None. Pass A_bin to train_model().")
                loss = loss + config.lambda_lap * laplacian_smoothness(yhat, A_bin)
            
            loss.backward()
            opt.step()
            tot += loss.item()
        train_mae = tot / max(1,len(loaders["train"]))
        
        model.eval()
        with torch.no_grad():
            def eval_on(split):
                tot = 0.0
                for batch in loaders[split]:
                    x = batch["x"].to(device)
                    y = batch["y"].to(device)
                    m = batch["mask"].to(device)
                    yhat = model(x)
                    tot += masked_mae(yhat, y, m).item()
                return tot / max(1,len(loaders[split]))
            val_mae = eval_on("val")
        print(f"epoch {epoch:03d}  train_mae={tot/max(1,len(loaders['train'])):.4f}  val_mae={val_mae:.4f}")

        hist["epoch"].append(epoch)
        hist["train"].append(train_mae)
        hist["val"].append(val_mae)

        # ---- NEW: epoch hook (can mutate loaders) ----
        if epoch_hook is not None:
            epoch_hook(epoch, model, loaders)

        if val_mae < best["val"] - 1e-6:
            best = {"val": val_mae, "state": model.state_dict(), "epoch": epoch}
            patience = config.patience
        else:
            patience -= 1
            if patience <= 0:
                print(f"early stop at epoch {epoch}, best epoch {best['epoch']} val_mae={best['val']:.4f}")
                break
        

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.history = hist
    return model


def evaluate(model, loader, device) -> Dict[str,float]:
    model.eval()
    mae_sum = rmse_sum = m_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            yhat = model(x)
            err = (yhat - y)
            mae_sum += (err.abs() * m).sum().item()
            rmse_sum += ((err**2)*m).sum().item()
            m_sum += m.sum().item()
    mae = mae_sum / (m_sum + 1e-8)
    rmse = math.sqrt(rmse_sum / (m_sum + 1e-8))
    return {"MAE": mae, "RMSE": rmse}


def _rollout_preds_over_timeline(model, X, M, exog, H_in, H_out, device):
    """
    Returns Xhat: [T,E] aligned to the original timeline.
    We use horizon-0 (next-step) predictions for each sliding window.
    """
    model.eval()
    T, E = X.shape
    Xhat = np.zeros((T, E), dtype=np.float32)

    ds_all = preprocessing.WindowedEdgeDataset(X, M, exog, H_in, H_out)
    loader_all = DataLoader(ds_all, batch_size=128, shuffle=False, drop_last=False,
                            pin_memory=str(device).startswith("cuda"))

    t0 = H_in  # first original index represented by the first sample
    offset = 0
    with torch.no_grad():
        for batch in loader_all:
            xb = batch["x"].to(device)
            yb = model(xb).cpu().numpy()     # [B,H_out,E]
            B = yb.shape[0]
            # horizon 0 = one-step ahead
            Xhat[t0 + offset : t0 + offset + B, :] = yb[:, 0, :]
            offset += B
    return Xhat

def make_prediction_driven_hook(
    X, M, times, E, cfg, weather_df, A_bin,
    slices, loaders,
    *,
    lags_self=(1,2,3,6,12),
    lags_neigh=(1,2,3),
    ema_alpha=0.7,
    fill_weights=(0.5, 0.3, 0.2),   # (EMA, neighbor, pred)  <-- now includes preds
    current_exog_container=None     # dict holding {"arr": exog}; if None we'll compute basic
):
    """
    Returns epoch_hook(epoch, model, loaders) that:
      - rolls out preds to Xhat
      - rebuilds exog with Xhat_prev blended in
      - recreates train/val loaders with the updated exog
    """
    # prepare a mutable holder for current exog used in rollout
    if current_exog_container is None:
        current_exog_container = {"arr": features.add_exogenous(
            times=times, E=E, config=cfg, weather_df=weather_df,
            X=X, M=M, A_bin=A_bin, use_graph_fill=True,
            lags_self=lags_self, lags_neigh=lags_neigh,
            ema_alpha=ema_alpha, fill_weights=(0.7, 0.3, 0.0), Xhat_prev=None
        )}

    def epoch_hook(epoch, model, loaders_dict):
        # 1) rollout predictions using *current* exog
        Xhat = _rollout_preds_over_timeline(
            model, X, M, current_exog_container["arr"], cfg.H_in, cfg.H_out, cfg.device
        )

        # 2) rebuild exog with prediction-driven fill
        exog_new = features.add_exogenous(
            times=times, E=E, config=cfg, weather_df=weather_df,
            X=X, M=M, A_bin=A_bin, use_graph_fill=True,
            lags_self=lags_self, lags_neigh=lags_neigh,
            ema_alpha=ema_alpha, fill_weights=fill_weights,  # uses predictions now
            Xhat_prev=Xhat
        )

        # keep it for the next epoch's rollout
        current_exog_container["arr"] = exog_new

        # 3) recreate train/val datasets & loaders (test unchanged)
        for split in ("train", "val"):
            s = slices[split]
            ds = preprocessing.WindowedEdgeDataset(
                X[s], M[s], exog_new[s], cfg.H_in, cfg.H_out
            )
            loaders_dict[split] = DataLoader(
                ds,
                batch_size=cfg.batch_size,
                shuffle=(split == "train"),
                drop_last=True,
                pin_memory=str(cfg.device).startswith("cuda")
            )

        print(f"[hook] Rebuilt exog with preds at epoch {epoch} "
              f"(weights={fill_weights}); loaders updated.")

    return epoch_hook


def laplacian_smoothness(yhat: torch.Tensor, A_bin_np: np.ndarray) -> torch.Tensor:
    """
    yhat: [B,H,E] predictions; A_bin_np: [E,E] 0/1 adjacency in numpy.
    Penalizes (y_i - y_j)^2 over neighboring directed edges.
    """
    A = torch.from_numpy(A_bin_np).to(yhat.device, non_blocking=True).bool()
    diff = yhat[..., None, :] - yhat[..., :, None]   # [B,H,E,E]
    sq = (diff**2)[..., A]
    return sq.mean()
