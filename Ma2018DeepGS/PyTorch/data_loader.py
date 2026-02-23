import numpy as np
import matplotlib.pyplot as plt
import os
import rdata

def load_data(save_path="saved_models"):
    trainMat = np.load("data/train_matrix.npy")
    trainPheno = np.load("data/train_pheno.npy")
    validMat = np.load("data/valid_matrix.npy")
    validPheno = np.load("data/valid_pheno.npy")
    markerImage = (1, n_markers)  # update based on your data!
    return trainMat, trainPheno, validMat, validPheno, markerImage

def load_wheat_data(save_path="saved_models",rngSeed=0):
    converted = rdata.read_rda("../data/wheat_example.rda")
    Markers = converted['wheat_example']['Markers'].values
    y = converted['wheat_example']['y'].values
    cvSampleList = cvSampleIndex(len(y),10)

    cvIdx = 0
    trainIdx = cvSampleList[cvIdx]['trainIdx']
    testIdx = cvSampleList[cvIdx]['testIdx']
    trainMat = Markers[trainIdx,]
    trainPheno = y[trainIdx]

    rng = np.random.default_rng(rngSeed)
    index = rng.choice(len(trainIdx),int(len(trainIdx)*0.1),replace=False)
    mask = np.ones_like(trainIdx, dtype=bool)
    mask[index] = False # set the chosen 10% of indices to False
    validMat = trainMat[~mask]
    validPheno = trainPheno[~mask]
    trainMat = trainMat[mask,]
    trainPheno = trainPheno[mask]

    n_markers = Markers.shape[1]
    markerImage = (1, n_markers)  # update based on your data!
    return trainMat, trainPheno, validMat, validPheno, markerImage


########################generate train idx and test idx ##########################
#' @title Generate Sample Indices for Training Sets and Testing Sets
#' @description  This function generates indices for samples in training and testing sets for performing the N-fold cross validation experiment.
#' @param sampleNum  The number of samples needed to be partitioned into training and testing sets.
#' @param cross  The fold of cross validation.
#' @param seed  An integer used as the seed for data partition. The default value is 1.
#' @param randomSeed  Logical variable. The default value is FALSE.
#' @return 
#' A list and each element including $trainIdx, $testIdx and $cvIdx.
#' 
#' $trainIdx  The index of training samples.
#' 
#' $testIdx   The index of testing samples.
#' 
#' $cvIdx     The index of cross validation.
#' @author Chuang Ma, Zhixu Qiu, Qian Cheng and Wenlong Ma
#' @export
#' @examples
#'#' ## Load example data ##
#' data(wheat_example)
#' ## 5-fold cross validation
#' b <- cvSampleIndex(sampleNum = 2000, cross = 5, seed = 1)

# get sample idx for training and testing
def cvSampleIndex(sampleNum, cv = 5, rngSeed = 1):
    resList = []  
    # leave-one-out
    if( cv == sampleNum ):
        vec = np.arange(sampleNum)
        for i in np.arange(sampleNum):
            mask = np.ones_like(vec, dtype=bool)
            mask[i] = False
            resList.append({'trainIdx': vec[mask], 'testIdx': i, 'cvIdx': i})
    else:
        #random samples 
        rng = np.random.default_rng(rngSeed)
        index = rng.choice(sampleNum,sampleNum,replace=False)
        step = int(np.floor( sampleNum/cv ))
        
        for i in np.arange(cv):
            start = step*i
            end = start + step
            if (i == cv): 
                end = sampleNum
            mask = np.ones_like(index, dtype=bool)
            mask[start:end] = False
            testIdx = index[start:end]
            trainIdx = index[mask]
            resList.append({'trainIdx': trainIdx, 'testIdx': testIdx, 'cvIdx': i})
    # names(resList) <- paste0("cv",1:cross)
    return(resList)


def generate_synthetic_data(n_markers=50,rng_seed=0,save_path="saved_models"):
    rng = np.random.default_rng(rng_seed)
    n_train = 400
    n_valid = 40
    # n_markers = 784
    # n_markers = 50
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

    # Create output directory if needed
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/data_validation.png", dpi=150, bbox_inches="tight")
    plt.close()

    return trainMat, trainPheno, validMat, validPheno, markerImage