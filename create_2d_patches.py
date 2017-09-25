import sys
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import os
import glob
import cPickle as pickle
from PIL import Image
from sklearn.model_selection import train_test_split
import SimpleITK as sitk




raw_image_path = './data/raw/*/'
candidates_file = './data/candidates.csv'

mode = 'val'
image_size = 512
patch_size = 140

pickles_dir = './data/pickles/'
patches_dir = './data/patches/'+str(patch_size)+'/'

class CTScan(object):
    def __init__(self, filename=None,label=None, nodule_coords=None, path=None):
        """
        Args
        -----
        filename: .mhd filename
        coords: coordinates to crop around
        ds: data structure that contains CT header data like resolution etc
        path: path to directory with all the raw data
        """
        self.filename = filename
        self.nodule_coords = nodule_coords
        self.meta_header = None
        self.image = None
        self.label = label
        self.path = path

    def reset_coords(self, nodule_coords):
        self.nodule_coords = nodule_coords

    def read_mhd_image(self):
        """
        Reads mhd data
        """
        mhdpath = glob.glob(self.path + self.filename + '.mhd')
        self.meta_header = sitk.ReadImage(mhdpath[0])  # mhdpath is just an array containing the path
        self.image = sitk.GetArrayFromImage(self.meta_header)

    def get_voxel_coords(self):
        """
        Converts cartesian to voxel coordinates
        """
        origin = self.meta_header.GetOrigin()
        resolution = self.meta_header.GetSpacing()

        voxel_coords = [np.absolute(self.nodule_coords[j] - origin[j]) / resolution[j] for j in
                        range(len(self.nodule_coords))]

        return tuple(voxel_coords)

    def get_image(self):
        """
        Returns axial CT slice
        """
        return self.image

    def random_crop(self, cropped_image_size):

        x, y, z = self.get_voxel_coords()

        sz1 = int(image_size // 2)
        sz2 = int(cropped_image_size // 2)

        diff = sz1 - sz2
        (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))

        cropped_image = self.image[int(z), (sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h)]
        return cropped_image

    def extract_patch_around_nodule(self, patch_size):
        x, y, z = self.get_voxel_coords()
        width = patch_size

        start_y = int(y - width / 2)
        end_y = int(y + width / 2)

        start_x = int(x - width / 2)
        end_x = int(x + width / 2)

        if start_y < 0:
            start_y = 0
            end_y = width
        elif end_y > image_size:
            start_y = image_size - width
            end_y = image_size
        else:
            pass

        if start_x < 0:
            start_x = 0
            end_x = width
        elif end_x > image_size:
            start_x = image_size - width
            end_x = image_size
        else:
            pass

        subImage = self.image[int(z), int(start_y):int(end_y), int(start_x):int(end_x)]

        return subImage

    def get_patch(self, patch_size):

        self.read_mhd_image()

        if(int(self.label) == 0):
            return self.random_crop(patch_size)

        if(int(self.label) == 1):
            return self.extract_patch_around_nodule(patch_size)

    def normalizePlanes(self, npzarray):
        """
        Copied from SITK tutorial converting Houndsunits to grayscale units
        """
        maxHU = 400.
        minHU = -1000.
        npzarray = (npzarray - minHU) / (maxHU - minHU)
        npzarray[npzarray > 1] = 1.
        npzarray[npzarray < 0] = 0.
        return npzarray

    def save_image(self, filename, patch_size):
        """
        Saves cropped CT image
        """
        image = self.get_patch(patch_size)
        image = self.normalizePlanes(image)

        print 'image shape: ' + str(image.shape)

        Image.fromarray(image * 255).convert('L').save(filename)


def create_data(idx, outDir, X_data, Y_data, patch_size):

    # CTScan(filename,nodule_coords,raw_path)
    scan = CTScan(np.asarray(X_data.loc[idx])[0], Y_data.loc[idx], np.asarray(X_data.loc[idx])[1:], raw_image_path)

    outfile = outDir + str(idx) + '.jpg'

    print 'saving image: ' + str(idx)

    scan.save_image(outfile, patch_size)


def do_test_train_split(csvfile):
    # test train splitting
    candidates = pd.read_csv(csvfile)  # cols are seriesuid, coordX, coordY, coordZ, class

    positives = candidates[candidates['class'] == 1].index  # pos candidates index
    negatives = candidates[candidates['class'] == 0].index  # neg candidates index

    print 'positive candidates: ' + str(len(positives))  # num of pos: 1351
    print 'negative candidates: ' + str(len(negatives))  # num of neg: 549714

    # take negative samples as equal number of positives
    negative_sampled = np.random.choice(negatives, len(positives), replace=False)
    indexes = np.append(positives, negative_sampled)    #merge pos and negs
    np.random.shuffle(indexes)  #randomly shuffle

    candidatesDf = candidates.iloc[list(indexes)]  # keep positives and sampled negative data only

    X = candidatesDf.iloc[:, :-1]  # X contains all but last "class" column
    Y = candidatesDf.iloc[:, -1]  # last "class" column is y

    print 'splitting into train test and val'

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=10, shuffle=True)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.20, random_state=10, shuffle=True)

    # total: 2702
    # train: 1836
    # test: 406
    # val: 460

    print 'train size: ' + str(len(X_train))
    print 'test size: ' + str(len(X_test))
    print 'val size: ' + str(len(X_val))

    print 'pickling..'
    train_set = (X_train, Y_train)
    test_set = (X_test, Y_test)
    val_set = (X_val, Y_val)

    #each pickle contains corresponding image file name, coordinates and label
    pickle.dump(train_set, open('./data/trainset.pickle', 'wb'))
    pickle.dump(test_set, open('./data/testset.pickle', 'wb'))
    pickle.dump(val_set, open('./data/valset.pickle', 'wb'))

def create_patches(mode):

    input_file = pickles_dir+mode+'set.pickle'
    output_dir = patches_dir+mode+'/patch_'

    if os.path.isfile(input_file):
        pass
    else:
        do_test_train_split(candidates_file)

    X_data, Y_data = pickle.load(open(input_file, 'rb'))

    print 'total ' + mode + ' data: ' + str(len(X_data))

    #width = 224
    #Parallel(n_jobs=3)(delayed(create_data)(idx, output_dir, X_data, 224) for idx in X_data.index)

    for idx in X_data.index:
        create_data(idx, output_dir, X_data, Y_data, patch_size)




if __name__ == "__main__":
    create_patches(mode)    #'train', 'test', or 'val'

