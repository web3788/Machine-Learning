import numpy as np
from data_utils import load_CIFAR10
from svm import SVM
import matplotlib.pyplot as plt


def visualize_image(X_train, y_train):
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 8
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()


def visualize_loss(loss_history):
    plt.plot(loss_history)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()


def preprocess_dataset():
    cifar10_dir = 'D:\Python38\WORKS\CODE\svm\cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    visualize_image(X_train, y_train)
    input('Press any key to perform cross-validation...')

    num_train = 48500
    num_val = 1500
    val_indices = range(num_train, num_train + num_val)
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    X_train = X_train[:num_train]
    y_train = y_train[:num_train]

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    mean_image = np.mean(X_train, axis=0)
    X_train = X_train - mean_image
    X_test = X_test - mean_image
    X_val = X_val - mean_image

    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])

    return X_train, y_train, X_test, y_test, X_val, y_val


def auto_get_parameters(X_train, y_train, X_val, y_val):
    learning_rates = [1e-7, 5e-5]
    regularization_strengths = [5e4, 1e5]

    best_parameters = None
    best_val = -1

    for lr in learning_rates:
        for reg_strength in regularization_strengths:
            svm = SVM()
            svm.train(X_train, y_train, reg_strength, 1, lr, 200, 1500, True)
            y_pred = svm.predict(X_val)
            acc_val = np.mean(y_val == y_pred)
            if best_val < acc_val:
                best_val = acc_val
                best_parameters = (lr, reg_strength)

    print('OK! The best validation accuracy achieved during cross-validation is: %f' % best_val)
    return best_parameters


def get_svm_model(parameters, X, y):
    svm = SVM()
    loss_history = svm.train(X, y, parameters[1], 1, parameters[0], 200, 1500, True)
    visualize_loss(loss_history)
    input('Enter any key to predict...')
    return svm


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, X_val, y_val = preprocess_dataset()
    best_parameters = auto_get_parameters(X_train, y_train, X_val, y_val)
    svm = get_svm_model(best_parameters, X_train, y_train)
    y_pred = svm.predict(X_test)
    print('Accuracy achieved during cross-validation: %f' % (np.mean(y_pred == y_test)))