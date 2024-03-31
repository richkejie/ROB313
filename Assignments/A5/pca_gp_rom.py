import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve

def load_cyl2d_dataset(print_details=False):
    """
    Loads the cylinder flow dataset.

    Inputs:
        print_details : (bool, optional) whether to print details about the dataset

    Outputs:
        x_train, x_valid, x_test, y_train, y_valid, y_test
    """
    with np.load('cylinder2d.npz') as data:
        u = data['u']
        v = data['v']
        xdim = data['x']
        ydim = data['y']
        tdim = data['t']

        # for time series dataset, feature-target pairs are (time, state)
        # even though state at a given time is 160x80x2, we flatten it to a 25600-long vector
        x = tdim
        y = np.reshape(np.stack((u, v), axis=3), (tdim.shape[0], -1))

        # divide the dataset into 1000 for training, 200 for validation, 300 for testing
        x_train = x[:1001]
        y_train = y[:1001]

        x_valid = x[1001:1201]
        y_valid = y[1001:1201]

        x_test = x[1201:1501]
        y_test = y[1201:1501]

    if print_details:
        print("Cylinder Flow Dataset")

        print("xdim = %d" % xdim.shape)
        print("ydim = %d" % ydim.shape)
        print("tdim = %d" % tdim.shape)

        print("d = %d" % x_train.shape[1])
        print("n_train = %d" % x_train.shape[0])
        print("n_valid = %d" % x_valid.shape[0])
        print("n_test = %d" % x_test.shape[0])

    return x_train, x_valid, x_test, y_train, y_valid, y_test, xdim, ydim

def find_pca_matrix(y_train, z_d):
    """
    Finds the matrix that encodes/decodes a dataset via PCA.

    Inputs:
        y_train : dataset to be encoded
        z_d : dimension of encoded state

    Outputs:
        pca_matrix : the PCA mode matrix, and 
        pca_vector : the PCA mean vector, where np.matmul(y_train - pca_vec, pca_matrix) = z_train
    """
    """ YOUR CODE HERE """
    pass

def sqexp_kernel(x, z, theta=1, variance=1.):
    """ one-dimensional squared exponential kernel with lengthscale theta """
    return variance * np.exp(-np.square(x - z.T) / theta)

def matern_kernel(x, z, theta=1, variance=1.):
    """ one-dimensional matern kernel with lengthscale theta """
    """ YOUR CODE HERE """
    pass

def gp_prediction(x, y, x_test, kernel, noise_var):
    """ Computes posterior mean and cov at x_test given data x, y """
    """ THIS IS CURRENTLY FOR SCALAR TARGETS ONLY """
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    x_test = x_test.reshape((-1, 1))

    N = x.shape[0]
    N_test = x_test.shape[0]

    C = cho_factor(kernel(x, x) + noise_var*np.identity(N))

    mu_pred = kernel(x_test, x).dot(cho_solve(C, y))
    
    cov_pred = (
        kernel(x_test, x_test) + noise_var*np.identity(N_test)
        - kernel(x_test, x).dot(cho_solve(C, kernel(x, x_test)))
    )
    return mu_pred, cov_pred

def gp_evidence(x, y, kernel, noise_var):
    """ Computes the GP log marginal likelihood """
    """ THIS IS CURRENTLY FOR SCALAR TARGETS ONLY """
    N = x.shape[0]

    C = cho_factor(kernel(x, x) + noise_var*np.identity(N))

    log_evidence = (
        0.5*y.T.dot(cho_solve(C, y))
        - np.sum(np.log(np.diag(C[0])))
        - 0.5*N*np.log(2*np.pi)
    )

    return log_evidence

if __name__ == "__main__":
    # loading data
    x_train, x_valid, x_test, y_train, y_valid, y_test, xdim, ydim = load_cyl2d_dataset()

    # state is currently flattened, keep track of true shape for plotting
    state_shape = [ydim.shape[0], xdim.shape[0], 2]

    # define pca dimension
    z_d = 4

    # plot state at last time step of training set
    f, axarr = plt.subplots(1, 2)
    img = axarr[0].imshow(np.reshape(y_train[-1], state_shape)[:,:,0])
    f.colorbar(img, ax=axarr[0])
    img = axarr[1].imshow(np.reshape(y_train[-1], state_shape)[:,:,1])
    f.colorbar(img, ax=axarr[1])
    plt.show()

    # do pca
    pca_matrix, pca_vec = find_pca_matrix(y_train, z_d)

    # encode flow state
    z_train = np.matmul(y_train - pca_vec, pca_matrix)
    z_valid = np.matmul(y_valid - pca_vec, pca_matrix)
    z_test = np.matmul(y_test - pca_vec, pca_matrix)

    # do type-ii inference for kernel hyperparameters
    """ YOUR CODE HERE """

    # do gp prediction over validation and test sets
    """ YOUR CODE HERE """

    # decode predictions
    y_valid_pred_mu = np.matmul(z_valid_pred_mean, pca_matrix) + pca_vec
    y_test_pred_mu = np.matmul(z_test_pred_mean, pca_matrix) + pca_vec

    # plot prediction at final time step
    """ YOUR CODE HERE """