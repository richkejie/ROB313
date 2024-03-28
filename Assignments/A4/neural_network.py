import autograd.numpy as np
from autograd.scipy.special import logsumexp
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam
import os
import sys
import matplotlib.pyplot as plt

# for confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# path to data
sys.path.append(os.path.join(os.getcwd(), "../../../data"))
from data_utils import load_dataset


def init_randn(m, n, rs=npr.RandomState(0)):
    """ init mxn matrix using small random normal
    """
    return 0.1 * rs.randn(m, n)


def init_xavier(m, n, rs=npr.RandomState(0)):
    """ TODO: init mxn matrix using Xavier intialization
    """
    epsilon = 1/np.sqrt(m) # m is number of input units, m = D
    return epsilon * rs.randn(m,n)


def init_net_params(layer_sizes, init_fcn, rs=npr.RandomState(0)):
    """ inits a (weights, biases) tuples for all layers using the intialize function (init_fcn)"""
    return [
        (init_fcn(m, n), np.zeros(n))  # weight matrix  # bias vector
        for m, n in zip(layer_sizes[:-1], layer_sizes[1:])
    ]


def relu(x):
    return np.maximum(0, x)


def neural_net_predict(params, inputs):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = relu(outputs)
    return outputs - logsumexp(outputs, axis=1, keepdims=True)


def mean_log_like(params, inputs, targets):
    """ TODO: return the log-likelihood / the number of inputs 
    """
    N = inputs.shape[0] # number of inputs
    log_probs = neural_net_predict(params, inputs) # normalized class log-probabilities

    # inputs is (N,D)
    # targets is (N,K)
    # log_probs is (N,K)
    outer_sum = 0
    for i in range(N):
        inner_sum = targets[i].dot(log_probs[i])
        outer_sum += inner_sum

    return outer_sum / N
    # raise NotImplementedError("mean_log_like function not completed (see Q3).")


def accuracy(params, inputs, targets):
    """ return the accuracy of the neural network defined by params
    """
    target_class = np.argmax(targets, axis=1)
    predicted_class = np.argmax(neural_net_predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)


if __name__ == "__main__":
    # ----------------------------------------------------------------------------------
    # EXAMPLE OF HOW TO USE STARTER CODE BELOW
    # ----------------------------------------------------------------------------------
    npr.seed(100)

    # loading data
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("mnist_small")

    # initializing parameters
    # below are original, starter code parametres
    # layer_sizes = [784, 200, 10]
    # params = init_net_params(layer_sizes, init_randn)
    
    # setting up training parameters
    # below are original, starter code training parameters
    # num_epochs = 5
    # learning_rate = 1e-1
    # batch_size = 256

    # these are my tuned parameters
    layer_sizes = [784, 200, 50, 10] # added another hidden layer with 50 neurons/units
    params = init_net_params(layer_sizes, init_xavier) # switch to xavier initialization

    num_epochs = 20 # 20
    learning_rate = 0.005 # smaller eta --> this greatly improves results, starter code eta was way too big...
    batch_size = 256 # same

    # Constants for batching
    num_batches = int(np.ceil(len(x_train) / batch_size))
    rind = np.arange(len(x_train))
    npr.shuffle(rind)

    def batch_indices(iter):
        idx = iter % num_batches
        return rind[slice(idx * batch_size, (idx + 1) * batch_size)]

    # Define training objective
    def objective(params, iter):
        # get indices of data in batch
        idx = batch_indices(iter)
        return -mean_log_like(params, x_train[idx], y_train[idx])

    # Get gradient of objective using autograd.
    objective_grad = grad(objective)

    # print("     Epoch     |    Train accuracy  |       Test accuracy  ") # test accuracy is actually validation accuracy...
    print("     Epoch     |    Train accuracy  | Validation accuracy  ")

    # Dictionary to store train/val history # added accuracy fields # add parameters fields
    opt_history = {
        "train_nll": [],
        "val_nll": [],

        "train_acc": [],
        "val_acc": [],
        "test_acc": [],

        "params": []
    }

    def callback(params, iter, gradient):
        if iter % num_batches == 0:
            # record training & val. accuracy every epoch
            opt_history["train_nll"].append(-mean_log_like(params, x_train, y_train))
            opt_history["val_nll"].append(-mean_log_like(params, x_valid, y_valid))
            train_acc = accuracy(params, x_train, y_train)
            val_acc = accuracy(params, x_valid, y_valid)
            
            # add train, valid, test_acc
            test_acc = accuracy(params, x_test, y_test)
            opt_history["train_acc"].append(train_acc)
            opt_history["val_acc"].append(val_acc)
            opt_history["test_acc"].append(test_acc)

            # save parameters
            opt_history["params"].append(params)

            print("{:15}|{:20}|{:20}".format(iter // num_batches, train_acc, val_acc))

    # We will optimize using Adam (a variant of SGD that makes better use of
    # gradient information).
    opt_params = adam(
        objective_grad,
        params,
        step_size=learning_rate,
        num_iters=num_epochs * num_batches,
        callback=callback,
    )

    # Report final training, validation, test accuracy
    print("Final results:")
    print("\tTrain accuracy: {}".format(opt_history["train_acc"][-1]))
    print("\tValidation accuracy: {}".format(opt_history["val_acc"][-1]))
    print("\tTest accuracy: {}".format(opt_history["test_acc"][-1]))

    # Plotting the train and validation negative log-likelihood
    plt.figure()
    plt.plot(opt_history["train_nll"], label="train_nll")
    plt.plot(opt_history["val_nll"], label="val_nll")
    plt.ylabel("Negative Mean Log-Likelihood")
    plt.xlabel("Epoch #")
    plt.title("Training and Validation Loss")
    plt.legend(loc='best')
    plt.show()

    # Plot the confusion matrix for the validation set
    target_class = np.argmax(y_valid, axis=1)
    predicted_class = np.argmax(neural_net_predict(opt_history["params"][-1], x_valid), axis=1)
    conf_matrix = confusion_matrix(target_class,
                                   predicted_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.show()
