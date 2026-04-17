import os
import cv2
import pandas as pd
import numpy as np

def load_data(data_dir, img_size=(64, 64)):
    """
    Loads images and their corresponding labels from a standard dataset directory structure
    where each subfolder represents a class label.
    """
    images = []
    labels = []
    filepaths = []
    
    classes = sorted(os.listdir(data_dir))
    for label in classes:
        label_folder = os.path.join(data_dir, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    images.append(img)
                    labels.append(label)
                    filepaths.append(img_path)
                    
    return np.array(images), np.array(labels), filepaths

def balance_dataset(filepaths, labels, samples_per_class=500, random_state=42):
    """
    Balances the dataset so that each class has exactly `samples_per_class` images.
    Returns the balanced filepath and label lists.
    """
    df = pd.DataFrame({'file_paths': filepaths, 'labels': labels})
    
    samples = []
    for category in df['labels'].unique():
        category_slice = df.query("labels == @category")
        # Ensure we don't try to sample more than what's available
        n_samples = min(samples_per_class, len(category_slice))
        samples.append(category_slice.sample(n_samples, replace=False, random_state=random_state))
        
    balanced_df = pd.concat(samples, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return balanced_df['file_paths'].tolist(), balanced_df['labels'].tolist()
