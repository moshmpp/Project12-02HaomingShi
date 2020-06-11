#-*- coding: utf8 -*-
from __future__ import print_function

import cv2 as cv
import numpy as np
import os
import simple_fourier_toolbox as fourier_toolbox


def decision_function(class_data):
    """A decision function for a class pattern in the minimum distance classifier"""
    mean_vector = np.mean(class_data, axis=0)
    def inner_function(x):
        result = x.dot(mean_vector)-0.5 * mean_vector.dot(mean_vector)
        return result
    return inner_function, mean_vector


def get_boundary_points(img):
    """Use the boundary following algorithm(Suzuki85) to get boudary points."""
    if len(img.shape) > 2:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    # Convert to binary image
    _, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
    # Retrieve only the extreme outer contours and store absolutely all the contour points
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # Sort the contours by the number of boundary points
    contours.sort(key=lambda x: len(x), reverse=True)
    return gray, contours


def get_fourier_descriptors(contours, contour_level, num_descriptors=18, invariant=True):
    """Get truncated Fourier Descriptors for the contour_level index in contours"""
    fourier_tool = fourier_toolbox.fourier_toolbox()
    contour_complex = fourier_tool.contour_to_complex(contours, contour_level)
    fourier_desc = np.fft.fft(contour_complex)
    if invariant:
        # make invariant to rotation and boundary starting point
        fourier_desc = fourier_tool.make_rotation_sp_invariant(fourier_desc)
        # make invariant to scale
        fourier_desc = fourier_tool.make_scale_invariant(fourier_desc)
        # make invariant to translation
        fourier_desc = fourier_tool.make_translation_invariant(fourier_desc)
    # truncate fourier descriptors
    fourier_subset = fourier_tool.get_low_frequencies(fourier_desc, num_descriptors)
    return fourier_subset
    

def get_samples(orig_vec, num_samples):
    """Based on vector `orig_vec`, generate `num_samples` samples by adding guassion noise."""
    num_features = len(orig_vec)
    vec = np.absolute(orig_vec)
    # mean and standard deviation (i.e. the maximum component of the vector divided by 10)
    mu, sigma = 0, np.amax(vec)/10.0
    gaussian_noise = np.random.normal(mu, sigma, (num_samples, num_features))
    samples = orig_vec + gaussian_noise
    return samples
    

def get_fourier_features(img, num_descriptors=18, contour_level=0):
    """Get fourier descriptor features with the specified number of features"""
    _, contours = get_boundary_points(img)
    fourier_desc = get_fourier_descriptors(contours, contour_level, num_descriptors)
    return fourier_desc
    

def load_leaf_shapes(dirname, traindata_percent=0.8, extract_features=get_fourier_features):
    """Load Leaf Shape Database, split into training set and test set by traindata_percent"""
    # Map class string to integer
    classes_map = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'g': 5, 'h': 6, 'i': 7, 'j': 8, 'k': 9, 'l': 10}
    traindata, trainlabels = [], []
    testdata, testlabels = [], []
    # Traverse leaf gray images
    subdirs = os.listdir(dirname)
    for subdir in subdirs:
        subdir = os.path.join(dirname, subdir)
        filenames = os.listdir(subdir)
        num_files = len(filenames)
        # Get training set and test set
        num_train = int(round(num_files * traindata_percent))
        for i, name in enumerate(filenames):
            path = os.path.join(subdir, name)
            try:
                class_id = classes_map[name[0]]
            except Exception as e:
                print('%s not found in classes' % name)
                continue
            img = cv.imread(path)
            features = extract_features(img)
            if i < num_train:
                traindata.append(features)
                trainlabels.append(class_id)
            else: 
                testdata.append(features)
                testlabels.append(class_id)
    traindata = np.array(traindata)
    trainlabels = np.array(trainlabels)
    testdata = np.array(testdata)
    testlabels = np.array(testlabels)
    return traindata, trainlabels, testdata, testlabels


def process_leaf_shapes():
    """Process leaf shape gray images in Leaf Shape Database, and split into training/test set, then
    classify with the minimum distance classifier"""
    print("\n"+"--" * 50+"\n")
    print("Process Leaf Shape Database ...\n")
    # Get training set and test set from Leaf Shape Database
    dirname = 'images/LeafShapesDatabase/'
    traindata, trainlabels, testdata, testlabels = load_leaf_shapes(dirname)
    # Learn decision functions from training set
    n_classes = len(set(trainlabels.flatten()))
    decision_func_list = [None] * n_classes
    mean_vectors = [None] * n_classes
    for label in range(1, n_classes+1):
        decision_func, mean_vector = decision_function(traindata[trainlabels==label])
        decision_func_list[label-1] = decision_func
        mean_vectors[label-1] = mean_vector
    # Classify test set
    results = []
    for decision_func in decision_func_list:
        if decision_func is None:
            continue
        res = [decision_func(vec) for vec in testdata]
        results.append(res)
    results = np.array(results).T
    # Choose the maximum of the output result of these decision functions
    final_res = np.argmax(results, axis=1) + 1
    print("Total class number: {}.".format(n_classes))
    print("training set number: {}.".format(len(trainlabels)))
    print("test set number: {}.".format(len(testlabels)))
    print("final result: {}".format(final_res))
    accuracy = sum(final_res == testlabels) / float(len(testlabels))
    print("accuracy: {}.".format(accuracy))


def main():
    print ("-" * 20 + " Project 12 " + "-" * 20)
    fname = 'images/Fig1218(airplanes).tif'
    # Read Image(Figure 12.18)
    img = cv.imread(fname)
    h, w = img.shape[:2]

    # Get boundary points of fig 12.18(a1) and (a2)
    gray_a1, contours_a1 = get_boundary_points(img[:h//2,:w//4])
    gray_a2, contours_a2 = get_boundary_points(img[:h//2,w//4:w//2])

    # Get truncated fourier descriptors 
    contour_level = 0
    num_descriptors = 18
    xa = get_fourier_descriptors(contours_a1, contour_level, num_descriptors, invariant=True)
    xb = get_fourier_descriptors(contours_a2, contour_level, num_descriptors, invariant=True)

    # Generate samples
    num_samples = 100
    trainset_xa = get_samples(xa, num_samples)
    trainset_xb = get_samples(xb, num_samples)
    testset_xa = get_samples(xa, num_samples)
    testset_xb = get_samples(xb, num_samples)

    # Train the minimum distance classifier and classify the testset data
    decision_func_xa, mean_vec_xa = decision_function(trainset_xa)
    decision_func_xb, mean_vec_xb = decision_function(trainset_xb)
    print("The mean vector of xa class pattern is {}.".format(mean_vec_xa))
    print("The mean vector of xb class pattern is {}.".format(mean_vec_xb))

    # Minimum distance classifier
    minimum_distance_classifier = lambda x : decision_func_xa(x) - decision_func_xb(x)

    # Establish the classifier recognition performance
    corrects_xa = sum(map(lambda x: minimum_distance_classifier(x) > 0, testset_xa))
    corrects_xb = sum(map(lambda x: minimum_distance_classifier(x) < 0, testset_xb))
    print("The correctly classified number for xa testset is {}.".format(corrects_xa))
    print("The correctly classified number for xb testset is {}.".format(corrects_xb))
    accuracy = (corrects_xa + corrects_xb)/ float(num_samples << 1)
    print("The accuracy is {}.".format(accuracy))


if __name__ == '__main__':
    main()
    process_leaf_shapes()
