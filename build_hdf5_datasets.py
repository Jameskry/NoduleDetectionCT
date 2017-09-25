import numpy as np
import cPickle as pickle
import h5py
from PIL import Image






mode = 'val'
patch_size = 140
target_image_size = 224
dataset_dir = './data/patches/'+str(patch_size)+'/'
pickle_dir = './data/pickles/'
hdf5_dir = dataset_dir+'hdf5/'


def one_hot_encode(Y_data, n_classes):
    Y_data = np.asarray(Y_data, dtype='int32')
    Y = np.zeros((len(Y_data), n_classes))
    Y[np.arange(len(Y_data)), Y_data] = 1.
    return Y

def build_hdf5_dataset(mode):
	X, Y = pickle.load(open(pickle_dir + mode + 'set.pickle','rb'))		#	X and Y are pandas Series
	filenames = X.index.to_series().apply(lambda x: dataset_dir + mode + '/patch_' + str(x) + '.jpg')

	img_files = filenames.values.astype(str)
	img_labels = Y.values.astype(int)

	n_classes = np.max(img_labels) + 1

	img_shape = (len(img_files), target_image_size, target_image_size, 1)  #grayscale images, so only 1 channel exists. mx224x224x1
	label_shape = (len(img_files), n_classes)	# mx2

	hdf5_dataset = h5py.File(hdf5_dir+mode+'.h5', 'w')
	hdf5_dataset.create_dataset('X', img_shape)
	hdf5_dataset.create_dataset('Y', label_shape)

	for i in range(len(img_files)):
		img = Image.open(img_files[i])
		width, height = img.size

		if width!=target_image_size or height!= target_image_size:
			img = img.resize((target_image_size, target_image_size))	#resize image by changing quality, not content

		img = img.convert('L')	# bcz Grayscale = True
		img.load() #untill this, file was open, but content wasnt loaded.
		img = np.asarray(img, dtype="float32")
		img = img/255 	#normalize
		img = img.reshape([img.shape[0], img.shape[1], 1])	# make 224x224 to 224x224x1

		hdf5_dataset['X'][i] = img
		hdf5_dataset['Y'][i] = one_hot_encode( [img_labels[i]], n_classes)

	hdf5_dataset.close()




if __name__ == '__main__':
	build_hdf5_dataset(mode)

'''
# Read data
X = pd.read_pickle('./data/'+mode + 'data.pickle')
y = pd.read_pickle('./data/'+mode + 'labels.pickle')


dataset_file = './data/'+mode + 'datalabels.txt'

filenames = X.index.to_series().apply(lambda x: './data/'+mode+ '/image_'+str(x)+'.jpg')


filenames = filenames.values.astype(str)
labels = y.values.astype(int)
data = np.zeros(filenames.size, dtype=[('col1', 'S36'), ('col2', int)])

data['col1'] = filenames
data['col2'] = labels

np.savetxt(dataset_file, data, fmt="%10s %d")#datapath to label map

output = './data/'+mode + 'dataset.h5'



build_hdf5_image_dataset(dataset_file, image_shape = (214, 214),
 mode ='file', output_path = output, categorical_labels = True, normalize = True,
 grayscale = True)

# Load HDF5 dataset
h5f = h5py.File('./data/'+ mode+ 'dataset.h5', 'r')
X_images = h5f['X']
Y_labels = h5f['Y'][:]

print X_images.shape
X_images = X_images[:,:,:].reshape([-1,512,512,1])
print X_images.shape
h5f.close()

h5f = h5py.File('./data/' + mode + '.h5', 'w')
h5f.create_dataset('X', data=X_images)
h5f.create_dataset('Y', data=Y_labels)
h5f.close()


'''