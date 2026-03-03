import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from deepgs_model import DeepGSModel, LightGSModel, LightGS1D
import os

from diagnostics import plot_actual_vs_predicted, evaluate_model, plot_both_vs_marker
import numpy as np
from datetime import datetime

import math
from copy import deepcopy


# trainMat, validMat (n_samples x n_markers)
# trainPheno, validPheno (n_samples x 1)
def train_deepGSModel( 
    trainMat, trainPheno, validMat, validPheno,
    markerImage, cnnFrame,
    device="cpu", eval_metric="mae",
    num_round=6000, batch_size=30, learning_rate=0.01,
    momentum=0.5, wd=1e-5, patience=600, verbose=True,
    save_path="saved_models"
):
    datetime_str = datetime.now().strftime("_%Y%m%d_%H%M%S")

    # Create output directory if needed
    os.makedirs(save_path, exist_ok=True)

    # reshape to NCHW
    H, W = markerImage
    trainMat4D = trainMat.reshape(-1, 1, H, W)
    validMat4D = validMat.reshape(-1, 1, H, W)

    train_tensor = torch.tensor(trainMat4D, dtype=torch.float32)
    valid_tensor = torch.tensor(validMat4D, dtype=torch.float32)
    y_train = torch.tensor(trainPheno, dtype=torch.float32).view(-1, 1)
    y_valid = torch.tensor(validPheno, dtype=torch.float32).view(-1, 1)

    model = DeepGSModel(cnnFrame, markerImage).to(device)
    # model = LightGSModel(markerImage).to(device)
    # model = LightGS1D(W).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=wd)
    criterion = nn.L1Loss() if eval_metric == "mae" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(    
        optimizer, mode="min", factor=0.5, patience=3)


    # Data loader
    train_dataset = torch.utils.data.TensorDataset(train_tensor, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    steps_no_improve = 0

    # --- Baseline MAE/MSE without any training ---
    model.eval()
    with torch.no_grad():
        base_pred = model(train_tensor.to(device))
        base_mae = torch.mean(torch.abs(base_pred.cpu() - y_train)).item()
    print(f"Baseline (untrained) Train MAE: {base_mae:.4g}")

    for epoch in range(num_round):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            pred = model(Xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            pred_valid = model(valid_tensor.to(device))
            val_loss = criterion(pred_valid, y_valid.to(device)).item()

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}, val {eval_metric}: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            os.makedirs(save_path, exist_ok=True)
            best_model_path = os.path.join(save_path, f"best_model{datetime_str}.pth")
            torch.save(model.state_dict(), best_model_path)
            steps_no_improve = 0
        else:
            steps_no_improve += 1

        if steps_no_improve > patience:
            if verbose:
                print("Early stopping triggered.")
            break

        best_model.eval()
        with torch.no_grad():
            pred_train = best_model(train_tensor.to(device))
            train_loss = criterion(pred_train, y_train.to(device)).item()
            pred_valid = best_model(valid_tensor.to(device))
            val_loss = criterion(pred_valid, y_valid.to(device)).item()
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}, best val loss {eval_metric}: {best_loss:.4f}, train loss {eval_metric}: {train_loss:.4f}, val loss {eval_metric}: {val_loss:.4f}")
            print(torch.mean(torch.abs(pred_train-y_train.to(device))))

    # ----- Diagnostics after training -----

    # Load best model before returning
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model.")
    else:
        print("Can't load best model.")

    model.eval()
    # Train predictions
    train_true, train_pred = evaluate_model(
        model, trainMat, trainPheno, markerImage, device=device)

    # train_true, train_pred = evaluate_model(
    #     model, trainMat.T, trainPheno, device=device
    # )

    plot_actual_vs_predicted(
        train_true, train_pred,
        title="Train: Actual vs Predicted",
        save_path=f"{save_path}/train_actual_vs_predicted{datetime_str}.png"
    )

    # Validation predictions
    valid_true, valid_pred = evaluate_model(
        model, validMat, validPheno, markerImage, device=device)
    # valid_true, valid_pred = evaluate_model(
    #     model, validMat.T, validPheno, device=device
    # )
    plot_actual_vs_predicted(
        valid_true, valid_pred,
        title="Validation: Actual vs Predicted",
        save_path=f"{save_path}/valid_actual_vs_predicted{datetime_str}.png"
    )

    plot_both_vs_marker(trainMat[:,30],
        train_true, train_pred,
        title="Train: Pheno vs Marker",
        save_path=f"{save_path}/train_pheno_vs_marker{datetime_str}.png"
    )
    plot_both_vs_marker(validMat[:,30],
        valid_true, valid_pred,
        title="Validation: Pheno vs Marker",
        save_path=f"{save_path}/valid_pheno_vs_marker{datetime_str}.png"
    )

    print(f"Diagnostic plots saved to: {save_path}")

    return model



# trainMat, validMat (n_samples x n_markers)
# trainPheno, validPheno (n_samples x 1)
def train_deepGSModel2(
    trainMat, trainPheno, validMat, validPheno,
    markerImage, cnnFrame,
    device="cpu", eval_metric="mae",
    num_round=6000, batch_size=30, learning_rate=0.01,
    momentum=0.5, wd=1e-5, patience=600, verbose=True,
    save_path="saved_models"
):
    datetime_str = datetime.now().strftime("_%Y%m%d_%H%M%S")

    # Create output directory if needed
    os.makedirs(save_path, exist_ok=True)

    # -------------------------------------------------------------
    # 1. trainMat, validMat originally shape: (N_samples, N_markers)
    #    markerImage = (H, W) such that H*W == N_markers
    # -------------------------------------------------------------

    H, W = markerImage  # assuming markerImage = (H, W)
    assert trainMat.shape[1] == H * W, "Marker count must match H*W"

    # -------------------------------------------------------------
    # 3. Reshape into NCHW tensors (CPU) for DeepGS-style CNN
    # -------------------------------------------------------------
    trainMat4D = trainMat.reshape(-1, 1, H, W)
    validMat4D = validMat.reshape(-1, 1, H, W)

    # -------------------------------------------------------------
    # 4. Convert to torch tensors on CPU (MPS-friendly)
    # -------------------------------------------------------------
    # Use from_numpy to avoid copies when you already have NumPy arrays
    train_tensor = torch.from_numpy(trainMat4D).float()       # CPU
    valid_tensor = torch.from_numpy(validMat4D).float()       # CPU
    y_train = torch.from_numpy(trainPheno).float().view(-1, 1)   # CPU
    y_valid = torch.from_numpy(validPheno).float().view(-1, 1)   # CPU

    # --- 2) DataLoaders (train shuffled, valid not) ---
    train_loader = DataLoader(
        TensorDataset(train_tensor, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,        # you can increase if CPU allows
        pin_memory=False       # if device is CUDA
    )

    valid_loader = DataLoader(
        TensorDataset(valid_tensor, y_valid),
        batch_size=batch_size,  # keep the same if you use BatchNorm
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # --- 3) Model / Optimiser / Loss ---
    model = DeepGSModel(cnnFrame, markerImage).to(device)
    optimizer = optim.SGD(model.parameters(),
                        lr=learning_rate,
                        momentum=momentum,
                        weight_decay=wd)
    criterion = nn.L1Loss() if eval_metric == "mae" else nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    steps_no_improve = 0

    os.makedirs(save_path, exist_ok=True)
    best_model_path = os.path.join(save_path, f"best_model{datetime_str}.pth")

    # --- Sanity checks on data ---
    def stats(name, t):
        t = t.detach()
        print(f"{name}: dtype={t.dtype}, shape={tuple(t.shape)}")
        print(f"  finite={torch.isfinite(t).all().item()}  "
            f"nan={torch.isnan(t).any().item()}  inf={torch.isinf(t).any().item()}")
        print(f"  min={t.min().item():.4g} max={t.max().item():.4g} mean={t.mean().item():.4g} std={t.std().item():.4g}")

    stats("X_train", train_tensor)
    stats("y_train", y_train)
    stats("X_valid", valid_tensor)
    stats("y_valid", y_valid)

    # --- Baseline MAE/MSE without any training ---
    model.eval()
    with torch.no_grad():
        base_pred = model(train_tensor.to(device))
        base_mae = torch.mean(torch.abs(base_pred.cpu() - y_train)).item()
    print(f"Baseline (untrained) Train MAE: {base_mae:.4g}")

    for epoch in range(num_round):
        # ----- Train -----
        model.train()
        running_train_loss = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            # track epoch average correctly
            bs = xb.size(0)
            running_train_loss += loss.item() * bs
            n_train += bs

        train_loss_epoch = running_train_loss / max(1, n_train)

        # ----- Validate -----
        model.eval()
        running_val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pv = model(xb)
                l = criterion(pv, yb)
                bs = xb.size(0)
                running_val_loss += l.item() * bs
                n_val += bs

        val_loss = running_val_loss / max(1, n_val)

        # Optional: log RMSE even when criterion is MSELoss
        if eval_metric.lower() in {"rmse", "mse"} and isinstance(criterion, nn.MSELoss):
            log_val = math.sqrt(val_loss)
            log_train = math.sqrt(train_loss_epoch)
            metric_name = "RMSE"
        else:
            log_val = val_loss
            log_train = train_loss_epoch
            metric_name = "MAE" if isinstance(criterion, nn.L1Loss) else "MSE"

        if verbose and (epoch % 100 == 0 or epoch == num_round - 1):
            print(f"Epoch {epoch:4d} | Train {metric_name}: {log_train:.4f} | Val {metric_name}: {log_val:.4f}")

        # ----- Early stopping on VAL loss -----
        if val_loss < best_loss - 1e-12:   # small tolerance to avoid oscillation
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            torch.save(best_state, best_model_path)
            steps_no_improve = 0
        else:
            steps_no_improve += 1

        if steps_no_improve >= patience:
            if verbose:
                print(f"Early stopping (no improvement for {patience} epochs).")
            break

    # --- 4) Load the true best weights before returning/using ---
    if best_state is None and os.path.isfile(best_model_path):
        best_state = torch.load(best_model_path, map_location=device)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    import matplotlib.pyplot as plt

    def plot_predictions(model, loader, device, title):
        model.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pv = model(xb)
                preds.append(pv.cpu().numpy())
                trues.append(yb.cpu().numpy())

        preds = np.vstack(preds).flatten()
        trues = np.vstack(trues).flatten()

        # Scatter plot
        plt.figure(figsize=(6, 6))
        plt.scatter(trues, preds, s=12, alpha=0.6)
        plt.plot([trues.min(), trues.max()],
                [trues.min(), trues.max()],
                'r--', lw=2)

        plt.xlabel("True phenotype")
        plt.ylabel("Predicted phenotype")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ---- After training is done and best weights loaded ----

    # Build DataLoaders for plotting (validation loader already exists)
    train_plot_loader = DataLoader(
        TensorDataset(train_tensor, y_train),
        batch_size=batch_size,
        shuffle=False
    )

    # Training scatter
    plot_predictions(model, train_plot_loader, device,
                    "Training: Prediction vs True")

    # Validation scatter
    plot_predictions(model, valid_loader, device,
                    "Validation: Prediction vs True")


        # best_model.eval()
        # with torch.no_grad():
        #     pred_train = best_model(train_tensor.to(device))
        #     train_loss = criterion(pred_train, y_train.to(device)).item()
        #     pred_valid = best_model(valid_tensor.to(device))
        #     val_loss = criterion(pred_valid, y_valid.to(device)).item()
        # if verbose and epoch % 100 == 0:
        #     print(f"Epoch {epoch}, best val loss {eval_metric}: {best_loss:.4f}, train loss {eval_metric}: {train_loss:.4f}, val loss {eval_metric}: {val_loss:.4f}")
        #     print(torch.mean(torch.abs(pred_train-y_train)))

    # ----- Diagnostics after training -----

    # Load best model before returning
    # if os.path.exists(best_model_path):
    #     model.load_state_dict(torch.load(best_model_path, map_location=device))
    #     print("Loaded best model.")
    # else:
    #     print("Can't load best model.")

    for parameter in model.parameters():
        print(parameter.data.shape)
        print(parameter.data)

    model.eval()
    # Train predictions
    train_true, train_pred = evaluate_model(
        model, trainMat, trainPheno, markerImage, device=device)

    # print(trainMat.shape,validMat.shape)
    # # print(pred_train)
    # # print(train_pred)
    # # train_true, train_pred = evaluate_model(
    # #     model, trainMat.T, trainPheno, device=device
    # # )

    plot_actual_vs_predicted(
        train_true, train_pred,
        title="Train: Actual vs Predicted",
        save_path=f"{save_path}/train_actual_vs_predicted{datetime_str}.png"
    )

    # # Validation predictions
    # valid_true, valid_pred = evaluate_model(
    #     model, validMat, validPheno, markerImage, device=device)
    # # valid_true, valid_pred = evaluate_model(
    # #     model, validMat.T, validPheno, device=device
    # # )
    # plot_actual_vs_predicted(
    #     valid_true, valid_pred,
    #     title="Validation: Actual vs Predicted",
    #     save_path=f"{save_path}/valid_actual_vs_predicted{datetime_str}.png"
    # )

    # # print(trainMat.shape)
    # # print(trainMat[:,0,0,30].T)
    # # print(trainMatStore.shape)
    # # print(trainMatStore[:,30].T)
    # # print(train_true.T)
    # print(trainMat[:,30].shape,train_true.shape)
    # plot_both_vs_marker(trainMat[:,30],
    #     train_true, train_pred,
    #     title="Train: Pheno vs Marker",
    #     save_path=f"{save_path}/train_pheno_vs_marker{datetime_str}.png"
    # )
    # plot_both_vs_marker(validMat[:,30],
    #     valid_true, valid_pred,
    #     title="Validation: Pheno vs Marker",
    #     save_path=f"{save_path}/valid_pheno_vs_marker{datetime_str}.png"
    # )

    # print(f"Diagnostic plots saved to: {save_path}")

    return model



def to_device_safe(x, device):
    """
    Move tensors or nested structures to the device.
    non_blocking=True only when useful (CUDA + pinned CPU).
    """
    if isinstance(x, torch.Tensor):
        use_nb = (device.type == "cuda") and x.is_pinned()
        return x.to(device, non_blocking=use_nb)
    elif isinstance(x, (list, tuple)):
        return type(x)(to_device_safe(t, device) for t in x)
    elif isinstance(x, dict):
        return {k: to_device_safe(v, device) for k, v in x.items()}
    else:
        return x


def train_model(
    model,
    train_loader,
    valid_loader,
    device,
    num_epochs=200,
    patience=10,
    optimizer=None,
    scheduler=None,
    criterion=None,
    save_path="checkpoints",
    verbose=True,
    grad_clip=5.0
):
    """
    A unified train() function that works for ANY of your models:
        - DeepGSModel (2D)
        - LightGSModel (2D row-safe)
        - LightGS1D (1D CNN)
        - Future variants

    Requirements:
        model: nn.Module
        train_loader, valid_loader: DataLoader objects
        device: torch.device('cuda'), 'mps', or 'cpu'

    """

    os.makedirs(save_path, exist_ok=True)

    if criterion is None:
        criterion = torch.nn.MSELoss()

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(num_epochs):
        # ---- TRAINING ----
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb = to_device_safe(xb, device)
            yb = to_device_safe(yb, device)

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            bs = xb.size(0)
            train_loss_sum += loss.item() * bs
            n_train += bs

        train_loss = train_loss_sum / max(1, n_train)

        # ---- VALIDATION ----
        model.eval()
        val_loss_sum = 0.0
        n_valid = 0

        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = to_device_safe(xb, device)
                yb = to_device_safe(yb, device)

                pred = model(xb)
                loss = criterion(pred, yb)

                bs = xb.size(0)
                val_loss_sum += loss.item() * bs
                n_valid += bs

        val_loss = val_loss_sum / max(1, n_valid)

        # Scheduler step
        if scheduler is not None:
            scheduler.step(val_loss)

        # Logging
        if verbose and (epoch % 100 == 0 or epoch == num_epochs - 1):
            # If MSELoss, log RMSE
            if isinstance(criterion, torch.nn.MSELoss):
                train_metric = math.sqrt(train_loss)
                val_metric = math.sqrt(val_loss)
                metric_name = "RMSE"
            else:
                train_metric = train_loss
                val_metric = val_loss
                metric_name = "MAE"

            print(f"Epoch {epoch:4d} | Train {metric_name}: {train_metric:.4f} | "
                  f"Val {metric_name}: {val_metric:.4f}")

        # Early stopping
        if val_loss < best_loss - 1e-12:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}.")
            break

    # ---- Restore best weights ----
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_loss