import os
import argparse
import numpy as np
from src.dataset import prepare_datasets, GENRES
from src.model import create_model, save_model
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main(data_dir):
    # Prepare data
    print("Preparing datasets...")
    # Check if 'genres' is a subdir of data_dir, if so adjust
    # Also handle 'genres_original'
    if os.path.exists(os.path.join(data_dir, 'genres')):
        data_dir = os.path.join(data_dir, 'genres')
    elif os.path.exists(os.path.join(data_dir, 'genres_original')): # Handle GTZAN original structure
        data_dir = os.path.join(data_dir, 'genres_original')
        
    X_train, y_train, X_val, y_val, X_test, y_test, encoder = prepare_datasets(data_dir, val_size=0.1)
    
    # Combine train and val for sklearn (optional, but RF doesn't strictly need val set for stopping)
    X_train_full = np.concatenate((X_train, X_val))
    y_train_full = np.concatenate((y_train, y_val))
    
    print(f"Train shape: {X_train_full.shape}")
    print(f"Test shape: {X_test.shape}")
    
    # Create and train model
    print("Training Random Forest...")
    model = create_model()
    model.fit(X_train_full, y_train_full)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    
    # Save model
    save_model(model, 'music_genre_rf.joblib')
    # Save class names
    np.save('class_names.npy', encoder.classes_)
    
    # --- Visualizations ---
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from itertools import cycle
    
    # 1. Confusion Matrix
    try:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
        formatted_labels = [label.title() for label in encoder.classes_] # Title case for labels
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join('graph', 'confusion_matrix.png'))
        plt.close()
        print("Saved confusion_matrix.png to graph/ folder")
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

    # 2. ROC Curve
    try:
        # Binarize output
        y_test_bin = label_binarize(y_test, classes=range(len(encoder.classes_)))
        n_classes = y_test_bin.shape[1]
        
        plt.figure(figsize=(10, 8))
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
        
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(encoder.classes_[i], roc_auc))
    
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Multi-class')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join('graph', 'roc_curve.png'))
        plt.close()
        print("Saved roc_curve.png to graph/ folder")
    except Exception as e:
        print(f"Error plotting ROC curve: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Music Genre Classifier (Random Forest)')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to dataset directory')
    # Epochs argument removed as it's not relevant for RF in this context
    
    args = parser.parse_args()
    main(args.data_dir)
