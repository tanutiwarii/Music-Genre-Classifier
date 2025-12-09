import os
import argparse
import numpy as np
import joblib
from src.dataset import prepare_datasets
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

def save_model_and_scaler(model, scaler, encoder, output_dir='.'):
    joblib.dump(model, os.path.join(output_dir, 'best_model.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    np.save(os.path.join(output_dir, 'class_names.npy'), encoder.classes_)
    print(f"Best model and scaler saved to {output_dir}")

def main(data_dir):
    # Check data path
    if os.path.exists(os.path.join(data_dir, 'genres')):
        data_dir = os.path.join(data_dir, 'genres')
    elif os.path.exists(os.path.join(data_dir, 'genres_original')):
        data_dir = os.path.join(data_dir, 'genres_original')
        
    print("Preparing datasets...")
    X_train, y_train, X_val, y_val, X_test, y_test, encoder = prepare_datasets(data_dir, val_size=0.1)
    
    # Combine train/val
    X_train_full = np.concatenate((X_train, X_val))
    y_train_full = np.concatenate((y_train, y_val))
    
    # Scaling is crucial for SVM and helps Gradient Boosting
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    
    from sklearn.ensemble import HistGradientBoostingClassifier
    
    models = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [200],
                'max_depth': [20],
                'min_samples_split': [2],
                'n_jobs': [-1] # Use all cores
            }
        },
        # SVM is extremely slow on 10k+ samples with grid search.
        # We will disable SVM for this high-data run or use a simpler configuration.
        # Let's use HistGradientBoosting instead which is very fast.
        'HistGradientBoosting': {
            'model': HistGradientBoostingClassifier(random_state=42),
            'params': {
                'learning_rate': [0.1, 0.2],
                'max_iter': [100, 200],
                'max_depth': [None, 10]
            }
        }
    }
    
    best_overall_acc = 0
    best_overall_model = None
    best_overall_name = ""
    
    import matplotlib.pyplot as plt
    
    model_names = []
    accuracies = []
    
    print(f"\nComparing {len(models)} models with Hyperparameter Tuning...")
    print("-" * 50)
    
    for name, config in models.items():
        print(f"Training {name}...")
        grid = GridSearchCV(config['model'], config['params'], cv=3, n_jobs=-1, scoring='accuracy')
        grid.fit(X_train_scaled, y_train_full)
        
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"  Best Params: {grid.best_params_}")
        print(f"  Test Accuracy: {acc*100:.2f}%")
        
        model_names.append(name)
        accuracies.append(acc * 100)
        
        if acc > best_overall_acc:
            best_overall_acc = acc
            best_overall_model = best_model
            best_overall_name = name
            
    print("-" * 50)
    print(f"WINNER: {best_overall_name} with {best_overall_acc*100:.2f}% Accuracy")
    
    # Plot Comparison Graph
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['#4c72b0', '#55a868', '#c44e52'][:len(model_names)])
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model comparison for Music Genre Classification', fontsize=14)
    plt.ylim(0, 100)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
                 
    if not os.path.exists('graph'):
        os.makedirs('graph')
        
    output_path = os.path.join('graph', 'model_comparison.png')
    plt.savefig(output_path)
    print(f"Saved comparison graph to {output_path}")
    
    # Check if we should save
    # Note: We save as 'music_genre_rf.joblib' (legacy name) or update predict.py to load 'best_model.joblib'
    # Let's save as 'best_model.joblib' and update predict.py separately to handle scaler.
    
    save_model_and_scaler(best_overall_model, scaler, encoder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced Training for Music Genre Classifier')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to dataset directory')
    
    args = parser.parse_args()
    main(args.data_dir)
