import numpy as np
import argparse
import os
import sys
import joblib
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.features import extract_features
from src.model import load_model

def predict(file_path, model_path='best_model.joblib', scaler_path='scaler.joblib', class_names_path='class_names.npy'):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    try:
        model = load_model(model_path)
    except FileNotFoundError:
        print(f"Error: Model {model_path} not found. Please train the model first.")
        return

    class_names = np.load(class_names_path, allow_pickle=True)
    
    # Extract features for segments
    # need to replicate split logic to get segments
    y, sr = librosa.load(file_path, sr=22050)
    segment_duration = 7
    samples_per_segment = int(segment_duration * sr)
    total_samples = len(y)
    
    segment_probs = []
    
    for start in range(0, total_samples, samples_per_segment):
        end = start + samples_per_segment
        if end > total_samples:
            break
            
        segment = y[start:end]
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
        cent = librosa.feature.spectral_centroid(y=segment, sr=sr)
        bw = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(segment)
        
        features = np.vstack([mfcc, chroma, cent, bw, rolloff, zcr])
        mean = np.mean(features, axis=1)
        std = np.std(features, axis=1)
        feature_vector = np.hstack((mean, std))
        
        # Reshape
        feature_vector = feature_vector.reshape(1, -1)
        
        # Scale
        if scaler_path and os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            feature_vector = scaler.transform(feature_vector)
            
        # Predict proba for this segment
        probs = model.predict_proba(feature_vector)[0]
        segment_probs.append(probs)
        
    if not segment_probs:
        print("Audio too short.")
        return

    # Average probabilities across all segments (Soft Voting)
    avg_probs = np.mean(segment_probs, axis=0)
    
    predicted_index = np.argmax(avg_probs)
    predicted_genre = class_names[predicted_index]
    confidence = np.max(avg_probs)
    
    # Use avg_probs for plotting
    probs = avg_probs
    predicted_index = np.argmax(probs)
    predicted_genre = class_names[predicted_index]
    confidence = np.max(probs)
    
    print(f"File: {file_path}")
    print(f"Predicted Genre: {predicted_genre}")
    print(f"Confidence: {confidence:.2f}")

    # Ensure graph directory exists
    graph_dir = 'graph'
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    # 1. Prediction Probabilities Chart
    try:
        plt.figure(figsize=(10, 6))
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        sorted_classes = class_names[sorted_indices]
        
        plt.bar(sorted_classes, sorted_probs, color='skyblue')
        plt.xlabel('Genre')
        plt.ylabel('Probability')
        plt.title(f'Genre Prediction Probabilities\nTop: {predicted_genre} ({confidence:.2f})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        prob_plot_path = os.path.join(graph_dir, 'prediction_probabilities.png')
        plt.savefig(prob_plot_path)
        plt.close()
        print(f"Prediction graph saved to {prob_plot_path}")
        
    except Exception as e:
        print(f"Could not create probability plot: {e}")

    # 2. Spectrogram
    try:
        y, sr = librosa.load(file_path, duration=30)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram of {os.path.basename(file_path)}')
        plt.tight_layout()
        
        spec_plot_path = os.path.join(graph_dir, 'spectrogram.png')
        plt.savefig(spec_plot_path)
        plt.close()
        print(f"Spectrogram saved to {spec_plot_path}")
        
    except Exception as e:
        print(f"Could not create spectrogram: {e}")
    
    return predicted_genre, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Music Genre')
    parser.add_argument('file_path', type=str, help='Path to audio file')
    parser.add_argument('--model', type=str, default='music_genre_rf.joblib', help='Path to saved model')
    
    args = parser.parse_args()
    predict(args.file_path, args.model)
