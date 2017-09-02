import tflearn
import h5py

image_size = 100

# Load HDF5 dataset
h5f = h5py.File('./data/train30_60.h5', 'r')
X_train_images = h5f['X']#shape=(no_of_image*height*width)
crop_size = (X_train_images.shape[1]-image_size)/2
X_train_images = X_train_images[:, crop_size:-crop_size, crop_size:-crop_size]
X_train_images = X_train_images.reshape([-1, X_train_images.shape[1], X_train_images.shape[2], 1])#shape=(no_of_image*height*width*num_of_channels)

Y_train_labels = h5f['Y']


h5f2 = h5py.File('./data/val5_10.h5', 'r')
X_val_images = h5f2['X']
crop_size = (X_val_images.shape[1]-image_size)/2
X_val_images = X_val_images[:, crop_size:-crop_size, crop_size:-crop_size]
X_val_images = X_val_images.reshape([-1, X_val_images.shape[1], X_val_images.shape[2], 1])#shape=(no_of_image*height*width*num_of_channels)

Y_val_labels = h5f2['Y']
#print X_train_images.shape
#print X_val_images.shape

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
net = tflearn.input_data(shape=[None, X_train_images.shape[1], X_train_images.shape[2], X_train_images.shape[3]],
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
# Training
model = tflearn.DNN(net, checkpoint_path='model_resnet_cifar10',
                    max_checkpoints=10, tensorboard_verbose=3, tensorboard_dir='./logs/',
                    clip_gradients=0.)

model.fit(X_train_images, Y_train_labels, n_epoch=20, validation_set=(X_val_images, Y_val_labels),
          snapshot_epoch=False, show_metric=True, batch_size=20, shuffle=True,
          run_id='resnet_nodule')


model.save("./checkpoints/nodule-classifier.tfl")

h5f.close()
h5f2.close()

