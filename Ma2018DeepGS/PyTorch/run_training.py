from data_loader import load_data, generate_synthetic_data
from train_deepgs import train_deepGSModel
from config import cnnFrame
import torch


def main():

    # trainMat, trainPheno, validMat, validPheno, markerImage = load_data()
    trainMat, trainPheno, validMat, validPheno, markerImage = generate_synthetic_data()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        print ("MPS device not found.")

    model = train_deepGSModel(
        trainMat=trainMat,
        trainPheno=trainPheno,
        validMat=validMat,
        validPheno=validPheno,
        markerImage=markerImage,
        cnnFrame=cnnFrame,
        device=device,
        eval_metric="mae",
        num_round=6000,
        batch_size=32,
        learning_rate=0.001,
        patience=10000
    )

    print("Training complete.")
    print(model)


if __name__ == "__main__":
    main()