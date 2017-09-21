import os
import tensorflow as tf
import numpy as np
import h5py
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score


train_file = 'traindataset_5_10.h5'
val_file = 'val5_10.h5'
test_file = 'test.h5'

data_dir = './data/'
checkpoint_file = './checkpoints/model-600'

#data_dir = os.environ['SCRATCH']+'/Data/tamjid_data/'
#checkpoint_file = os.environ['SCRATCH']+'/Data/aowal_data/project/cxr/models/resnet152/runs/1491968515-nodule-42/checkpoints/model-600'

learning_rate = 0.001
batch_size = 3
num_epochs = 3
image_size = 224
evaluate_every = 2

def crop(images, cropped_image_size, random_crop = True):
    if images.shape[2]>cropped_image_size:
        sz1 = int(images.shape[2]//2)
        sz2 = int(cropped_image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        images = images[: ,(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h)]
    return images

def load_hdf5_data(filename, cropped_size, random_crop = True):
    with h5py.File(data_dir+filename, 'r') as h5f:
        X_train_images = h5f['X']
        X_train_images = crop(X_train_images, cropped_size, random_crop)
        Y_train_labels = h5f['Y'][()]

    return X_train_images, Y_train_labels



def make_2d_to_3d(images_2d):
    X = []
    for i, d in enumerate(images_2d):
        d3 = np.dstack((images_2d[i], np.zeros(images_2d[i].shape), np.zeros(images_2d[i].shape)))
        X.append(d3)
    X = np.array(X)
    return X

def get_metrics(Y_test_labels, label_predictions):

    cm = confusion_matrix(Y_test_labels[:,1], label_predictions[:,1])

    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]

    precision = 0.
    recall = 0.
    specificity = 0.

    if TP <= 0:
        precision = 0.
        recall = 0.
    if TN <= 0:
        specificity = 0.
    else:
        precision = TP * 1.0 / (TP + FP)
        recall = TP * 1.0 / (TP + FN)
        specificity = TN * 1.0 / (TN + FP)

    return precision, recall, specificity, cm

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]






with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    sess = tf.Session(config=session_conf)


    with sess.as_default():
        # Load the saved meta graph and restore variables

        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_file)

        #for op in tf.get_default_graph().get_operations():
        #    print tf.get_default_graph().get_operation_by_name(str(op.name)).outputs

        input_x = tf.get_default_graph().get_tensor_by_name("input_x:0")
        input_y = tf.get_default_graph().get_tensor_by_name("input_y:0")


        logits = tf.get_default_graph().get_tensor_by_name('BiasAdd:0')

        prob = tf.get_default_graph().get_tensor_by_name('prob:0')

        loss = tf.get_default_graph().get_tensor_by_name('Mean:0')

        global_step = tf.get_default_graph().get_tensor_by_name('global_step:0')
        #global_step = tf.Variable(0, trainable=False, name='global_step')



        classes = tf.argmax(input=logits, axis=1)
        correct_predictions = tf.equal(classes, tf.argmax(input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

        optimizer = tf.train.AdamOptimizer(learning_rate, name='AdamOpt')
        train_op = optimizer.minimize(loss, global_step= global_step)

        sess.run(tf.global_variables_initializer())

        train_x_1d, train_y = load_hdf5_data(train_file, image_size, random_crop=True)
        train_x = make_2d_to_3d(train_x_1d)


        batches = batch_iter(list(zip(train_x, train_y)), batch_size, num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            batch_x, batch_y = zip(*batch)

            cur_logits, cur_prob, cur_loss,cur_accuracy,cur_step, _ = sess.run([logits, prob, loss,accuracy, global_step, train_op], {input_x: batch_x , input_y: batch_y})
            print "Current step: "+str(cur_step)+" current loss: "+str(cur_loss)+ " current accuracy: " + str(cur_accuracy*100)

            if cur_step % evaluate_every == 0:
                val_x_1d, val_y = load_hdf5_data(val_file, image_size, random_crop=True)
                val_x = make_2d_to_3d(val_x_1d)
                val_logits, val_prob, val_loss, val_accuracy,val_step, _ = sess.run([logits, prob, loss, accuracy,global_step, train_op],
                                                                           {input_x: val_x, input_y: val_y})

                label_predictions = np.zeros_like(val_logits)
                label_predictions[np.arange(len(val_logits)), val_logits.argmax(1)] = 1

                precision, recall, specificity, cm = get_metrics(val_y, label_predictions)

                print "______________validation________________"
                print " step: " + str(val_step) + " loss: " + str(
                    val_loss) + " accuracy: " + str(val_accuracy * 100)
                print " precision: "+str(precision)+" recall:"+str(recall)+" TN:"+str(cm[0][0])+" TP:"+str(cm[1][1]) +"\n"


