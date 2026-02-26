import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted", save_path=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', lw=2)
    plt.xlabel("Actual phenotype")
    plt.ylabel("Predicted phenotype")
    plt.title(title)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show
    # plt.close()

def plot_both_vs_marker(marker, y_true, y_pred, title="Pheno vs Marker", save_path=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(marker, y_true, alpha=0.6)
    plt.scatter(marker, y_pred, alpha=0.6)
    plt.xlabel("Marker value")
    plt.ylabel("Phenotype")
    plt.title(title)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

# mat should be a genome matrix shape (n_samples x n_markers)
def evaluate_model(model, mat, pheno, markerImage, device="cpu"):
    H, W = markerImage

    print(mat.shape)

    # Match training reshape
    X = mat.reshape(-1, 1, H, W)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y_true = pheno

    print(X.shape)

    model.eval()
    with torch.no_grad():
        preds = model(X).cpu().numpy().flatten()

    return y_true, preds

# def evaluate_model(model, X, y, device="cpu"):
#     """Returns numpy arrays (y_true, y_pred)."""
#     model.eval()
#     with torch.no_grad():
#         preds = model(torch.tensor(X, dtype=torch.float32).to(device))
#     preds = preds.cpu().numpy().flatten()
#     return y, preds