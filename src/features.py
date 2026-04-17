import cv2
import numpy as np
from skimage.feature import hog

def extract_hog_features_from_images(images, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Extracts Histogram of Oriented Gradients (HOG) features from a NumPy array of images.
    Returns a 2D array and optionally the HOG visualization images.
    """
    hog_features = []
    for img in images:
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feature = hog(gray_image, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=False)
        hog_features.append(feature)
        
    return np.array(hog_features)

def extract_hog_features_from_paths(filepaths, img_size=(64, 64), pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Given a list of filepaths, loads them, converts to grayscale, and extracts HOG features.
    """
    hog_features = []
    for path in filepaths:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, img_size)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            feature = hog(gray_image, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=False)
            hog_features.append(feature)
            
    return np.array(hog_features)
