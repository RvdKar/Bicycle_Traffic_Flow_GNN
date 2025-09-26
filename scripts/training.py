from __future__ import annotations
import math
from typing import Dict
from tqdm.auto import tqdm
import torch
import torch.optim as optim
# from preprocessing import Config

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


def train_model(model, loaders, config: Config):
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