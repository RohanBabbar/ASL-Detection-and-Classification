import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from dataset import load_data, balance_dataset
from features import extract_hog_features_from_paths

def plot_confusion_matrix(y_true, y_pred, classes, title):
    conf_mat = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

def main(args):
    print(f"Loading data from {args.data_dir}...")
    images, labels, filepaths = load_data(args.data_dir, img_size=(64, 64))
    print(f"Total images found: {len(filepaths)}")
    
    if args.balance:
        print(f"Balancing dataset to {args.samples_per_class} per class...")
        filepaths, labels = balance_dataset(filepaths, labels, samples_per_class=args.samples_per_class)
        print(f"Balanced dataset size: {len(filepaths)}")
    else:
        filepaths, labels = list(filepaths), list(labels)
        
    print("Extracting HOG features... (This may take a while)")
    X = extract_hog_features_from_paths(filepaths, img_size=(64, 64))
    print(f"Feature matrix shape: {X.shape}")
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples. Testing set: {X_test.shape[0]} samples.\n")
    
    models = {
        "SVM": SVC(kernel='linear', random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"--- Training {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} Accuracy: {acc:.4f}\n")
        
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        if args.visualize:
            plot_confusion_matrix(y_test, y_pred, label_encoder.classes_, f'{name} Confusion Matrix')
            
    print("=== Final Results ===")
    for name, acc in results.items():
        print(f"{name}: {acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASL classification models using HOG features.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory (e.g., 'data/asl_dataset').")
    parser.add_argument("--balance", action="store_true", help="Whether to balance the dataset before training.")
    parser.add_argument("--samples_per_class", type=int, default=500, help="Number of samples to keep per class if balancing.")
    parser.add_argument("--visualize", action="store_true", help="Plot confusion matrices after training.")
    
    args = parser.parse_args()
    main(args)
