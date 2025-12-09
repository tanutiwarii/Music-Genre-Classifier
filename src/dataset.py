import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
import librosa
# Add src to path to import features
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import extract_features

# GTZAN dataset constants
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']



def split_and_extract(file_path, label, X, y, sample_rate=22050, segment_duration=7, n_mfcc=13):
    try:
        # Load full audio
        audio, sr = librosa.load(file_path, sr=sample_rate)
        
        # Calculate samples per segment
        samples_per_segment = int(segment_duration * sr)
        total_samples = len(audio)
        
        # Loop through segments
        for start in range(0, total_samples, samples_per_segment):
            end = start + samples_per_segment
            if end > total_samples:
                break
                
            segment = audio[start:end]
            
            # Extract features for this segment
            # 1. MFCCs
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
            chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
            cent = librosa.feature.spectral_centroid(y=segment, sr=sr)
            bw = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(segment)
            
            features = np.vstack([mfcc, chroma, cent, bw, rolloff, zcr])
            
            # Aggregate: mean and std -> (n_features * 2,)
            mean = np.mean(features, axis=1)
            std = np.std(features, axis=1)
            feature_vector = np.hstack((mean, std))
            
            X.append(feature_vector)
            y.append(label)
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def load_data(dataset_path, sample_rate=22050):
    X = []
    y = []
    
    print(f"Loading data from {dataset_path} with 7s segmentation...")
    
    if not os.path.exists(dataset_path):
         raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    for genre in GENRES:
        genre_path = os.path.join(dataset_path, genre)
        if not os.path.isdir(genre_path):
            continue
            
        print(f"Processing genre: {genre}")
        for filename in os.listdir(genre_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(genre_path, filename)
                split_and_extract(file_path, genre, X, y, sample_rate=sample_rate)

    return np.array(X), np.array(y)

def prepare_datasets(dataset_path, test_size=0.2, val_size=0.2):
    """
    Loads data and splits into train, validation, and test sets.
    """
    X, y = load_data(dataset_path)
    
    # Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size / (1 - test_size), random_state=42, stratify=y_train)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, encoder
