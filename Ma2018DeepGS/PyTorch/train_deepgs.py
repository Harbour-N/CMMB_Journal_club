import torch
import torch.nn as nn
import torch.optim as optim
from deepgs_model import DeepGSModel
import os

from diagnostics import plot_actual_vs_predicted, evaluate_model, plot_both_vs_marker
import numpy as np
from datetime import datetime

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

    train_tensor = torch.tensor(trainMat4D, dtype=torch.float32).to(device)
    valid_tensor = torch.tensor(validMat4D, dtype=torch.float32).to(device)
    y_train = torch.tensor(trainPheno, dtype=torch.float32).view(-1, 1).to(device)
    y_valid = torch.tensor(validPheno, dtype=torch.float32).view(-1, 1)

    model = DeepGSModel(cnnFrame, markerImage).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=wd)
    criterion = nn.L1Loss() if eval_metric == "mae" else nn.MSELoss()

    # Data loader
    train_dataset = torch.utils.data.TensorDataset(train_tensor, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    steps_no_improve = 0

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
            print(torch.mean(torch.abs(pred_train-y_train)))

    # ----- Diagnostics after training -----

    # Load best model before returning
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model.")
    else:
        print("Can't load best model.")

    for parameter in model.parameters():
        print(parameter.data.shape)
        print(parameter.data)

    model.eval()
    # Train predictions
    train_true, train_pred = evaluate_model(
        model, trainMat, trainPheno, markerImage, device=device)

    print(trainMat.shape,validMat.shape)
    # print(pred_train)
    # print(train_pred)
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

    # print(trainMat.shape)
    # print(trainMat[:,0,0,30].T)
    # print(trainMatStore.shape)
    # print(trainMatStore[:,30].T)
    # print(train_true.T)
    print(trainMat[:,30].shape,train_true.shape)
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