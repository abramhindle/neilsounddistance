import argparse
import wave
import numpy as np
import csv
from python_speech_features import mfcc
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler

# Step 1: Read large wave file in chunks and compute MFCC
def process_wav_in_chunks(wav_file, chunk_duration_sec=1, mfcc_features=13, output_csv="mfcc_output.csv"):
    # Open the wave file
    wav = wave.open(wav_file, 'r')
    framerate = wav.getframerate()
    num_channels = wav.getnchannels()
    bytes_per_sample = wav.getsampwidth()
    
    # Number of samples in one second
    chunk_size = framerate * chunk_duration_sec
    print(f'chunk_size:{chunk_size}')
    mfcc_matrix = []
    row = 0
    # Open the CSV file to write the MFCCs per second
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        while True:
            # Read the chunk of audio data
            raw_data = wav.readframes(chunk_size)
            
            if not raw_data:
                break
            
            # Convert raw audio data to numpy array
            audio_data = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, num_channels)
            
            # Compute MFCCs for the audio data (use only one channel if stereo)
            mfcc_features_data = mfcc(audio_data[:, 0], nfft=2048, samplerate=framerate, numcep=mfcc_features)
            
            # # Average the MFCC across the chunk (to get a single MFCC vector per second)
            mfcc_avg = np.mean(mfcc_features_data, axis=0)
            mfcc_matrix.append(mfcc_avg)
            
            # Write the MFCC vector to the CSV file
            writer.writerow(mfcc_avg)

    wav.close()
    print(mfcc_matrix[-1].shape)
    print(mfcc_matrix[-2].shape)
    return np.array(mfcc_matrix)

# Step 2: Apply UMAP on the MFCC data
def apply_umap(mfcc_matrix, n_neighbors=17, min_dist=0.01, n_components=2):
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    umap_data = umap_model.fit_transform(mfcc_matrix)
    return umap_data

# Step 3: Apply HDBSCAN clustering
def apply_hdbscan(umap_data, min_cluster_size=5):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = clusterer.fit_predict(umap_data)
    return cluster_labels

# Step 4: Save the clustering results to a CSV
def save_clusters_to_csv(cluster_labels, output_csv="cluster_output.csv"):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        for label in cluster_labels:
            writer.writerow([label])

# Main workflow
def process_audio_and_cluster(wav_file, mfcc_output_csv="mfcc_output.csv", cluster_output_csv="cluster_output.csv", min_dist=0.1, do_umap=True):
    # Process the audio file and compute MFCCs
    mfcc_matrix = process_wav_in_chunks(wav_file, output_csv=mfcc_output_csv)
    
    # Normalize MFCC data
    scaler = StandardScaler()
    mfcc_matrix_normalized = scaler.fit_transform(mfcc_matrix)
    
    if do_umap:
        # Apply UMAP
        umap_data = apply_umap(mfcc_matrix_normalized, min_dist=min_dist)
        # umap_data = apply_umap(mfcc_matrix, min_dist=min_dist)
    else:
        umap_data = mfcc_matrix_normalized

    # Apply HDBSCAN
    cluster_labels = apply_hdbscan(umap_data)
    # cluster_labels = apply_hdbscan( mfcc_matrix )
    
    # Save cluster results
    save_clusters_to_csv(cluster_labels, output_csv=cluster_output_csv)

# Argument parsing function
def parse_args():
    parser = argparse.ArgumentParser(description="Process a wave file to extract MFCCs, apply UMAP, and cluster using HDBSCAN.")
    parser.add_argument('wav_file', type=str, help="Path to the wave file")
    parser.add_argument('--mfcc_output', type=str, default="mfcc_output.csv", help="Path to the output CSV file for MFCCs")
    parser.add_argument('--cluster_output', type=str, default="cluster_output.csv", help="Path to the output CSV file for cluster labels")
    parser.add_argument('--mindist', type=float, default=0.1,help="umap min distance")
    parser.add_argument('--noumap', action='store_true',help="Don't use umap")
    
    return parser.parse_args()

# Main execution function
if __name__ == "__main__":
    args = parse_args()
    process_audio_and_cluster(args.wav_file, args.mfcc_output, args.cluster_output, args.mindist,do_umap=(not args.noumap))
