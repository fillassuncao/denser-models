import sys
import getopt
import keras
from keras import backend
from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score
import fashion_mnist
from load_data import load_mnist_rotated, load_mnist_background, load_svhn, load_rectangles, load_rectangles_images, load_mnist_rotated_background

def augmentation(x):
    pad_size = 4

    h, w, c = WIDTH, HEIGHT, NUM_CHANNELS
    pad_h = h + 2 * pad_size
    pad_w = w + 2 * pad_size

    pad_img = np.zeros((pad_h, pad_w, c))
    pad_img[pad_size:h+pad_size, pad_size:w+pad_size, :] = x

    # Randomly crop and horizontal flip the image
    top = np.random.randint(0, pad_h - h + 1)
    left = np.random.randint(0, pad_w - w + 1)
    bottom = top + h
    right = left + w
    if np.random.randint(0, 2):
        pad_img = pad_img[:, ::-1, :]

    aug_data = pad_img[top:bottom, left:right, :]

    return aug_data


def load_data():
    SCALE = 255.0

    if DATASET == 'cifar-10':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    elif DATASET == 'cifar-100':
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    elif DATASET == 'fashion-mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = np.reshape(x_train, (-1, 28, 28, 1))
        x_test = np.reshape(x_test, (-1, 28, 28, 1))
    elif DATASET == 'mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, (-1, 28, 28, 1))
        x_test = np.reshape(x_test, (-1, 28, 28, 1))
    elif DATASET == 'mnist-rotated':
        x_train, y_train, x_test, y_test = load_mnist_rotated()
        x_train = np.reshape(x_train, (-1, 28, 28, 1))
        x_test = np.reshape(x_test, (-1, 28, 28, 1))
        SCALE = 1.0
    elif DATASET == 'mnist-background':
        x_train, y_train, x_test, y_test = load_mnist_background()
        x_train = np.reshape(x_train, (-1, 28, 28, 1), order='F')
        x_test = np.reshape(x_test, (-1, 28, 28, 1), order='F')
        SCALE = 1.0
    elif DATASET == 'mnist-rotated-background':
        x_train, y_train, x_test, y_test =load_mnist_rotated_background()
        x_train = np.reshape(x_train, (-1, 28, 28, 1), order='F')
        x_test = np.reshape(x_test, (-1, 28, 28, 1), order='F')
        SCALE = 1.0
    elif DATASET == 'svhn':
        x_train, y_train, x_test, y_test = load_svhn()
        x_train = np.reshape(x_train, (-1, 32, 32, 3))
        x_test = np.reshape(x_test, (-1, 32, 32, 3))
        SCALE = 1.0
    elif DATASET == 'rectangles':
        x_train, y_train, x_test, y_test = load_rectangles()
        x_train = np.reshape(x_train, (-1, 28, 28, 1))
        x_test = np.reshape(x_test, (-1, 28, 28, 1))
        SCALE = 1.0
    elif DATASET == 'rectangles-background':   
        x_train, y_train, x_test, y_test = load_rectangles_images()
        x_train = np.reshape(x_train, (-1, 28, 28, 1))
        x_test = np.reshape(x_test, (-1, 28, 28, 1))
        SCALE = 1.0

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= SCALE
    x_test /= SCALE

    x_mean = 0
    for x in x_train:
        x_mean += x
    x_mean /= len(x_train)
    x_train -= x_mean
    x_test -= x_mean

    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    dataset = {'x_train': x_train, 'y_train': y_train,
                'x_test': x_test, 'y_test': y_test}

    return dataset

def test(one_model=True):
    
    models = []
    predictions = []

    dataset = load_data()

    if one_model:
        for train_idx in xrange(NUM_TRAINS):
            print 'loading', train_idx
            models.append(load_model("%s/net_1/%s" % (TRAIN_DIR, TRAIN_FILENAME % train_idx), custom_objects={"backend": backend}))

    else:
        for train_idx in xrange(NUM_TRAINS*2):
            print 'loading', train_idx
            if train_idx < 5:
                models.append(load_model("%s/net_1/%s" % (TRAIN_DIR, TRAIN_FILENAME % train_idx), custom_objects={"backend": backend}))
            else:
                models.append(load_model("%s/net_9/%s" % (TRAIN_DIR, TRAIN_FILENAME % (train_idx-NUM_TRAINS)), custom_objects={"backend": backend}))



    for model in models:
        print 'predicting...'
        for _ in range(AUGMENT_TEST):
            x_test_augmented = np.array([augmentation(image) for image in dataset['x_test']])
            predictions.append(model.predict(x_test_augmented, batch_size=BATCH_SIZE, verbose=2))

    avg_prediction = np.average(predictions, axis=0)
    y_pred = np.argmax(avg_prediction, axis=1)
    y_true = np.argmax(dataset['y_test'], axis=1)
    accuracy = accuracy_score(y_true, y_pred)

    print accuracy

NUM_TRAINS = 5

AUGMENT_TEST = 100
TRAIN_FILENAME = 'train_%d.hdf5'

if __name__ == '__main__':
    options, remainder = getopt.getopt(sys.argv[1:], 'd:m', ['dataset=',
                                                             'multiple',
                                                            ])

    multiple = True
    DATASET = None
    BATCH_SIZE = 125
    
    for opt, arg in options:
        if opt in ('-d', '--dataset'):
            if arg == 'cifar-10':
                TRAIN_DIR = './CIFAR-10/'
                DATASET = 'cifar-10'
                NUM_CLASSES = 10
                WIDTH = 32
                HEIGHT = 32
                NUM_CHANNELS = 3
            elif arg == 'cifar-100':
                TRAIN_DIR = './CIFAR-100/'
                DATASET = 'cifar-100'
                NUM_CLASSES = 100
                WIDTH = 32
                HEIGHT = 32
                NUM_CHANNELS = 3
            elif arg == 'mnist':
                TRAIN_DIR = './MNIST/Standard/'
                DATASET = 'mnist'
                NUM_CLASSES = 10
                WIDTH = 28
                HEIGHT = 28
                NUM_CHANNELS = 1
            elif arg == 'mnist-rotated':
                TRAIN_DIR = './MNIST/Rotated/'
                DATASET = 'mnist-rotated'
                NUM_CLASSES = 10
                WIDTH = 28
                HEIGHT = 28
                NUM_CHANNELS = 1
            elif arg == 'mnist-background':
                TRAIN_DIR = './MNIST/Background/'
                DATASET = 'mnist-background'
                NUM_CLASSES = 10
                WIDTH = 28
                HEIGHT = 28
                NUM_CHANNELS = 1
            elif arg == 'mnist-rotated-background':
                TRAIN_DIR = './MNIST/Rotated+Background/'
                DATASET = 'mnist-rotated-background'
                NUM_CLASSES = 10
                WIDTH = 28
                HEIGHT = 28
                NUM_CHANNELS = 1
            elif arg == 'fashion-mnist':
                TRAIN_DIR = './FASHION-MNIST/'
                DATASET = 'fashion-mnist'
                NUM_CLASSES = 10
                WIDTH = 28
                HEIGHT = 28
                NUM_CHANNELS = 1
            elif arg == 'rectangles':
                TRAIN_DIR = './Rectangles/Standard/'
                DATASET = 'rectangles'
                NUM_CLASSES = 2
                WIDTH = 28
                HEIGHT = 28
                NUM_CHANNELS = 1
                BATCH_SIZE = 100
            elif arg == 'rectangles-background':
                TRAIN_DIR = './Rectangles/Background/'
                DATASET = 'rectangles-background'
                NUM_CLASSES = 2
                WIDTH = 28
                HEIGHT = 28
                NUM_CHANNELS = 1
                BATCH_SIZE = 100

        if opt in ('-m', '--multiple'):
            multiple = False

    if DATASET is None:
        print 'Invalid dataset!'

    else:
        test(multiple)

