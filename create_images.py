#!/usr/bin/env python

"""
Builds image data base as test, train, validatation datasets
Run script as python create_images.py $mode
where mode can be 'test', 'train', 'val'

"""

import sys

from joblib import Parallel, delayed

import numpy as np
import pandas as pd

import os
import glob

from PIL import Image

from sklearn.model_selection import train_test_split

import SimpleITK as sitk

raw_image_path = './data/raw/*/'
candidates_file = './data/candidates.csv'


class CTScan(object):


    def __init__(self, filename = None, nodule_coords = None, path = None):
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
        self.path = path

    def reset_coords(self, nodule_coords):

        self.nodule_coords = nodule_coords

    def read_mhd_image(self):
        """
        Reads mhd data
        """
        mhdpath = glob.glob(self.path + self.filename + '.mhd')
        self.meta_header = sitk.ReadImage(mhdpath[0]) #mhdpath is just an array containing the path
        self.image = sitk.GetArrayFromImage(self.meta_header)

    def get_voxel_coords(self):
        """
        Converts cartesian to voxel coordinates
        """
        origin = self.meta_header.GetOrigin()
        resolution = self.meta_header.GetSpacing()

        voxel_coords = [np.absolute(self.nodule_coords[j]-origin[j])/resolution[j] for j in range(len(self.nodule_coords))]

        return tuple(voxel_coords)
    
    def get_image(self):
        """
        Returns axial CT slice
        """
        return self.image
    
    def get_subimage(self, width):
        """
        Returns cropped image of requested dimensiona
        """
        self.read_mhd_image()
        x, y, z = self.get_voxel_coords()
        #subImage = self.image[int(z), int(y-width/2):int(y+width/2), int(x-width/2):int(x+width/2)]
        subImage = self.image[int(z), :, : ] #no subimage, take full image
        return subImage   
    
    def normalizePlanes(self, npzarray):
        """
        Copied from SITK tutorial converting Houndsunits to grayscale units
        """
        maxHU = 400.
        minHU = -1000.
        npzarray = (npzarray - minHU) / (maxHU - minHU)
        npzarray[npzarray>1] = 1.
        npzarray[npzarray<0] = 0.
        return npzarray
    
    def save_image(self, filename, width):
        """
        Saves cropped CT image
        """
        image = self.get_subimage(width)
        image = self.normalizePlanes(image)

        print 'image shape: '+str(image.shape)

        Image.fromarray(image*255).convert('L').save(filename)


def create_data(idx, outDir, X_data,  width = 50):
    '''
    Generates your test, train, validation images
    outDir = a string representing destination
    width (int) specify image size
    '''
    #CTScan(filename,nodule_coords,raw_path)
    scan = CTScan(np.asarray(X_data.loc[idx])[0], np.asarray(X_data.loc[idx])[1:], raw_image_path)

    outfile = outDir  +  str(idx)+ '.jpg'

    print 'saving image: '+str(idx)

    scan.save_image(outfile, width)

def do_test_train_split(filename):
    #test train splitting
    candidates = pd.read_csv(filename)

    positives = candidates[candidates['class']==1].index #pos candidates
    negatives = candidates[candidates['class']==0].index #neg candidates

    print 'positive candidates: '+str(len(positives))
    print 'negative candidates: ' + str(len(negatives))

    #neg are way more than pos. so we take samples from negatives.
    np.random.seed(42)
    negative_sampled = np.random.choice(negatives, len(positives)*5, replace = False)

    candidatesDf = candidates.iloc[list(positives)+list(negative_sampled)]

    X = candidatesDf.iloc[:,:-1]
    y = candidatesDf.iloc[:,-1]

    print 'splitting into train test and val'

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = 42)

    print 'train size: ' + str(len(X_train))
    print 'test size: ' + str(len(X_test))
    print 'val size: ' + str(len(X_val))

    print 'pickling..'

    X_train.to_pickle('./data/traindata.pickle')
    y_train.to_pickle('./data/trainlabels.pickle')
    X_test.to_pickle('./data/testdata.pickle')
    y_test.to_pickle('./data/testlabels.pickle')
    X_val.to_pickle('./data/valdata.pickle')
    y_val.to_pickle('./data/vallabels.pickle')


def main():
    if len(sys.argv) < 2:
        raise ValueError('1 argument needed. Specify if you need to generate a train, test or val set')
    else:
        mode = sys.argv[1]
        if mode not in ['train', 'test', 'val']:
            raise ValueError('Argument not recognized. Has to be train, test or val')

    inpfile = './data/'+mode + 'data.pickle'
    outDir = './data/'+mode + '/image_'

    if os.path.isfile(inpfile):
        pass
    else:
        do_test_train_split(candidates_file)
    X_data = pd.read_pickle(inpfile)

    print 'total '+mode+' data: '+str(len(X_data))

    Parallel(n_jobs = 3)(delayed(create_data)(idx, outDir, X_data,100) for idx in X_data.index)#100 width patches

if __name__ == "__main__":
    main()

        