import numpy as np
import matplotlib.pyplot as plt

def load_data():
    trainMat = np.load("data/train_matrix.npy")
    trainPheno = np.load("data/train_pheno.npy")
    validMat = np.load("data/valid_matrix.npy")
    validPheno = np.load("data/valid_pheno.npy")

    rng = np.random.default_rng(12345)
    n_train = 400
    n_valid = 40
    n_markers = 784
    n_markers = 50
    # n_markers = 2048

    allMat = rng.choice(2,(n_markers,n_train+n_valid))

    trainMat = rng.choice(2,(n_markers,n_train))
    validMat = rng.choice(2,(n_markers,n_valid))
    trainMat = allMat[:,0:n_train]
    validMat = allMat[:,n_train:]
    betas = np.zeros((1,n_markers))
    # betas[[10,20,30]] = 1
    active_markers = [30]
    active_markers = [10,20,30]
    betas[0,active_markers] = 1
    # betas[0][[10,20,30]]=1
    # trainPheno = trainMat @ betas + 0.0*rng.standard_normal((n_train,1),)
    # validPheno = validMat @ betas + 0.0*rng.standard_normal((n_valid,1),)
    
    # random numbers with a small perturbation across each row
    # trainMat = rng.random((1,n_train)) + 0.0*rng.standard_normal((n_markers,n_train))
    # validMat = rng.random((1,n_valid)) + 0.0*rng.standard_normal((n_markers,n_valid))
    # betas = np.ones((1,n_markers))
    trainPheno = betas @ trainMat + 0.01*rng.standard_normal((1,n_train),)
    validPheno = betas @ validMat + 0.01*rng.standard_normal((1,n_valid),)

    markerImage = (1, n_markers)  # update based on your data!

    plt.figure(figsize=(6, 6))
    plt.scatter(trainMat[active_markers[0],:], trainPheno, alpha=0.6, label='Train')
    plt.scatter(validMat[active_markers[0],:], validPheno, alpha=0.6, label='Validate')
    plt.xlabel("Marker value")
    plt.ylabel("Phenotype")
    plt.title('Data validation')
    plt.grid(True)
    plt.legend()

    plt.savefig('saved_models/data_validation.png', dpi=150, bbox_inches="tight")
    plt.close()

    return trainMat, trainPheno, validMat, validPheno, markerImage