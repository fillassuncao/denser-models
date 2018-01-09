import sys
import getopt
import keras
from keras import backend
from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score

def augmentation(x):
    pad_size = 4

    h, w, c = 32, 32, 3
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


def load_cifar(n_classes=100, test_size=7500):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    x_mean = 0
    for x in x_train:
        x_mean += x
    x_mean /= len(x_train)
    x_train -= x_mean
    x_test -= x_mean

    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    dataset = {'x_train': x_train, 'y_train': y_train,
                'x_test': x_test, 'y_test': y_test}

    return dataset

def test(dataset=10, one_model=True):
    
    models = []
    predictions = []

    if dataset == 10:
        for train_idx in xrange(NUM_TRAINS):
            models.append(load_model("%s/%s" % (CIFAR_10_DIR, TRAIN_FILENAME % train_idx), custom_objects={"backend": backend}))

    elif dataset == 100:
        if one_model:
            for train_idx in xrange(NUM_TRAINS):
                models.append(load_model("%s/net_1/%s" % (CIFAR_10_DIR, TRAIN_FILENAME % train_idx), custom_objects={"backend": backend}))
        else:
            for train_idx in xrange(NUM_TRAINS*2):
                if train_idx < 5:
                    models.append(load_model("%s/net_1/%s" % (CIFAR_10_DIR, TRAIN_FILENAME % train_idx), custom_objects={"backend": backend}))
                else:
                    models.append(load_model("%s/net_9/%s" % (CIFAR_10_DIR, TRAIN_FILENAME % train_idx), custom_objects={"backend": backend}))


    for model in models:
        for _ in range(AUGMENT_TEST):
            x_test_augmented = np.array([augmentation(image) for image in dataset['x_test']])
            predictions.append(model.predict(x_test_augmented, batch_size=125, verbose=2))

    avg_prediction = np.average(predictions, axis=0)
    y_pred = np.argmax(avg_prediction, axis=1)
    y_true = np.argmax(dataset['y_test'], axis=1)
    accuracy = accuracy_score(y_true, y_pred)

    print accuracy

NUM_TRAINS = 5
AUGMENT_TEST = 100
CIFAR_10_DIR = './CIFAR-10/'
CIFAR_100_DIR = './CIFAR-100/'
TRAIN_FILENAME = 'train_%d.hdf5'

if __name__ == '__main__':
    options, remainder = getopt.getopt(sys.argv[1:], 'd:m', ['dataset=',
                                                             'multiple',
                                                            ])

    dataset = None
    multiple = True
    
    for opt, arg in options:
        if opt in ('-d', '--dataset'):
            if arg is 'cifar-10':
                dataset = 10
            elif arg is 'cifar-100':
                dataset = 100

        if opt in ('-m', '--multiple'):
            one_model = False

    if dataset is None:
        print 'Invalid dataset!'

    else:
        test(dataset, multiple)

