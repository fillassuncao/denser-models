import scipy.io
import numpy as np

def load_mat_mnist_rotated(path):
    f_content = None

    with open(path, 'r') as f:
        f_content = f.readlines()

    x = list()
    y = list()
    if f_content is not None:
        for instance in f_content:
            instance_split = instance.rstrip().lstrip().replace('\n', '').split(' ')
            _class_ = int(float(instance_split[-1]))
            _image_ = map(float, instance_split[:-1])
            x.append(_image_)
            y.append(_class_)

    return x, y


def load_mat_mnist_rotated_background(path):
    f_content = None

    with open(path, 'r') as f:
        f_content = f.readlines()

    x = list()
    y = list()
    if f_content is not None:
        for instance in f_content:
            instance_split = instance.rstrip().lstrip().replace('\n', '').split('  ')
            _class_ = int(float(instance_split[-1]))
            _image_ = map(float, instance_split[:-1])
            x.append(_image_)
            y.append(_class_)

    return x, y


def load_mat_mnist_background(path):
    f_content = None

    with open(path, 'r') as f:
        f_content = f.readlines()

    x = list()
    y = list()
    if f_content is not None:
        for instance in f_content:
            instance_split = instance.rstrip().lstrip().replace('\n', '').split('   ')
            _class_ = int(float(instance_split[-1]))
            _image_ = map(float, instance_split[:-1])
            x.append(_image_)
            y.append(_class_)

    return x, y


def load_mat(path, limit):
    data = scipy.io.loadmat(path)

    x = np.rollaxis(data['X'], 3, 0)
    y = data['y']-1

    print x.shape, y.shape

    return x[:limit], y[:limit]


def load_mnist_rotated(n_classes=10):
    (x_train, y_train) = load_mat_mnist_rotated('./MNIST/Rotated/dataset/mnist_all_rotation_normalized_float_test.amat')
    (x_test, y_test) = load_mat_mnist_rotated('./MNIST/Rotated/dataset/mnist_all_rotation_normalized_float_train_valid.amat')

    return x_train, y_train, x_test, y_test


def load_mnist_background(n_classes=10):
    (x_train, y_train) = load_mat_mnist_background('./MNIST/Background/dataset/mnist_background_images_test.amat')
    (x_test, y_test) = load_mat_mnist_background('./MNIST/Background/dataset/mnist_background_images_train.amat')

    return x_train, y_train, x_test, y_test


def load_mnist_rotated_background(n_classes=10):
    (x_train, y_train) = load_mat_mnist_rotated_background('./MNIST/Rotated+Background/dataset/mnist_all_background_images_rotation_normalized_test.amat')
    (x_test, y_test) = load_mat_mnist_rotated_background('./MNIST/Rotated+Background/dataset/mnist_all_background_images_rotation_normalized_train_valid.amat')

    return x_train, y_train, x_test, y_test


def load_svhn(n_classes=10):
    (x_train, y_train) = load_mat('./SVHN/dataset/train_32x32.mat', 73250)
    (x_test, y_test) = load_mat('./SVHN/dataset/test_32x32.mat', 26000)

    return x_train, y_train, x_test, y_test


def load_rectangles(n_classes=2):
    (x_train, y_train) = load_mat_mnist_background('./Rectangles/Standard/dataset/rectangles_train.amat')
    (x_test, y_test) = load_mat_mnist_background('./Rectangles/Standard/dataset/rectangles_test.amat')
    
    return x_train, y_train, x_test, y_test


def load_rectangles_images(n_classes=2):
    (x_train, y_train) = load_mat_mnist_background('./Rectangles/Background/dataset/rectangles_im_train.amat')
    (x_test, y_test) = load_mat_mnist_background('./Rectangles/Background/dataset/rectangles_im_test.amat')
    
    return x_train, y_train, x_test, y_test