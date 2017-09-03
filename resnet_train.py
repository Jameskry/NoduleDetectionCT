import tflearn
import h5py
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import itertools
import matplotlib.pyplot as plt
import os



image_size = 100



def load_hdf5_data(filename):
    #with h5py.File(os.environ['SCRATCH']+'/Data/tamjid_data/train.h5', 'r') as h5f:
    with h5py.File(filename, 'r') as h5f:
        X_train_images = h5f['X']  # shape=(no_of_image*height*width)
        crop_size = (X_train_images.shape[1] - image_size) / 2
        X_train_images = X_train_images[:, crop_size:-crop_size, crop_size:-crop_size]
        X_train_images = X_train_images.reshape(
            [-1, X_train_images.shape[1], X_train_images.shape[2], 1])  # shape=(no_of_image*height*width*num_of_channels)

        Y_train_labels = h5f['Y'][()]

    return X_train_images, Y_train_labels


def get_network(X_input_images):

    # Real-time data preprocessing
    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_featurewise_zero_center(per_channel=True)

    # Real-time data augmentation
    img_aug = tflearn.ImageAugmentation()
    img_aug.add_random_flip_leftright()
    #img_aug.add_random_crop([32, 32], padding=4)


    # Residual blocks
    # 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
    n = 5


    # Building Residual Network
    net = tflearn.input_data(shape=[None, X_input_images.shape[1], X_input_images.shape[2], X_input_images.shape[3]],
                             data_preprocessing=img_prep,
                             data_augmentation=img_aug)
    net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
    net = tflearn.residual_block(net, n, 16)
    net = tflearn.residual_block(net, 1, 32, downsample=True)
    net = tflearn.residual_block(net, n-1, 32)
    net = tflearn.residual_block(net, 1, 64, downsample=True)
    net = tflearn.residual_block(net, n-1, 64)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)
    # Regression
    net = tflearn.fully_connected(net, 2, activation='softmax')
    adam = tflearn.Adam(0.001, beta1=0.9, beta2=0.999)
    net = tflearn.regression(net, optimizer=adam, loss='categorical_crossentropy')
    return net


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Purples):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    #plt.grid('off')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_metrics(Y_test_labels, label_predictions):

    cm = confusion_matrix(Y_test_labels[:,1], label_predictions[:,1])

    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]

    precision = TP*1.0/(TP+FP)
    recall = TP*1.0/(TP+FN)
    specificity = TN*1.0/(TN+FP)

    return precision, recall, specificity, cm

def predict():
    X_test_images, Y_test_labels = load_hdf5_data('./data/test5_10.h5')

    network = get_network(X_test_images)

    model = tflearn.DNN(network, checkpoint_path='./checkpoints/nodule-classifier.tfl', tensorboard_verbose=3)
    model.load("./checkpoints/nodule-classifier.tfl")

    prediction = model.predict(X_test_images)
    #classes = np.argmax(prediction, axis=1)

    #score = model.evaluate(X_test_images, Y_test_labels)

    label_predictions = np.zeros_like(prediction)
    label_predictions[np.arange(len(prediction)), prediction.argmax(1)] = 1
    precision, recall, specificity, cm = get_metrics(Y_test_labels, label_predictions)

    print precision, recall, specificity

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=['no-nodule', 'nodule'], title='Confusion matrix')
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    #plt.show()

def train():

    #initialization
    X_train_images, Y_train_labels = load_hdf5_data('./data/traindataset_5_10.h5')
    X_val_images, Y_val_labels = load_hdf5_data('./data/val5_10.h5')
    network = get_network(X_train_images)

    #train
    model = tflearn.DNN(network, checkpoint_path='model_resnet_nodule',
                        max_checkpoints=10, tensorboard_verbose=3, tensorboard_dir='./logs/',
                        clip_gradients=0.)

    model.fit(X_train_images, Y_train_labels, n_epoch=20, validation_set=(X_val_images, Y_val_labels),
              snapshot_epoch=False, show_metric=True, batch_size=20, shuffle=True,
              run_id='resnet_nodule')

    model.save("./checkpoints/nodule-classifier.tfl")

if __name__ == "__main__":
    predict()
    #train()




