import librosa
import numpy as np
import os

def extract_features(file_path, sample_rate=22050, duration=30, n_mfcc=13):
    """
    Extracts MFCC features from an audio file.
    
    Args:
        file_path (str): Path to the audio file.
        sample_rate (int): Sampling rate for audio loading.
        duration (int): Duration of audio to load in seconds.
        n_mfcc (int): Number of MFCCs to extract.
        
    Returns:
        np.array: MFCC features transposed (time_steps, n_mfcc) or None if failure.
    """
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
        
        # 1. MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # 2. Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        
        # 3. Spectral Centroid
        cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
        
        # 4. Spectral Bandwidth
        bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        
        # 5. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        
        # 6. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        
        # Stack all features
        # MFCC: (n_mfcc, time)
        # Chroma: (12, time)
        # Others: (1, time)
        features = np.vstack([mfcc, chroma, cent, bw, rolloff, zcr])
        
        # Return features transposed: (time, n_features)
        return features.T
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def save_features(features, labels, output_path):
    """
    Saves features and labels to a compressed numpy file.
    """
    np.savez_compressed(output_path, X=features, y=labels)
    print(f"Features saved to {output_path}")
