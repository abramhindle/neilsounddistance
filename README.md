# Compare the current second to the last ten seconds

python3 sounddistance.py yourwavefile

# Source

## Prompt

In python, read in a very long wave file that cannot fit in memory, and MFCC each second of the wavefile, write that matrix of MFCCs per second to a csv file, then compare that second against the past 10 seconds of MFCCs using cosine and euclidean distance, report those distances to a csv file

## Response

https://chatgpt.com/share/67168d42-ba88-8013-891e-1403bc9eb843

2024-10-21T11:21:21-06:00

MFCC UMAP HDBSCAN Processing
============================

October 21, 2024

This conversation may reflect the link creator's personalized data, which isn't shared and can meaningfully change how the model responds.

##### You said:

In python, read in a very long wave file that cannot fit in memory, and MFCC each second of the wavefile, write that matrix of MFCCs per second to a csv file, then apply UMAP to the rows of that matrix, apply hdbscan to cluster the normalized umap data and write out a new csv of cluster per second.

###### ChatGPT said:

ChatGPT

To process a very long wave file that cannot fit in memory, we can use a streaming approach. This approach will process the file in small chunks, applying the MFCC (Mel-frequency cepstral coefficients) transformation on each second of audio, writing those coefficients to a CSV file. After that, we'll apply UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction on the MFCCs, and then use HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) to cluster the UMAP output. Finally, we will save the cluster labels for each second in another CSV file.

Here's the step-by-step implementation:

1.  **Process the large audio file in chunks**: We'll use libraries such as `wave` to read the audio file in small chunks, `python_speech_features` to compute MFCCs, and `csv` to write the MFCCs.
2.  **Dimensionality reduction with UMAP**: Once we have the MFCCs for each chunk (per second), we apply UMAP to reduce their dimensionality.
3.  **Clustering with HDBSCAN**: After dimensionality reduction, we will cluster the UMAP-reduced data.
4.  **Write cluster results to a CSV file**: Save the cluster results (per second) to a CSV file.

We'll need these libraries:

-   `python_speech_features` for computing MFCCs.
-   `wave` or `librosa` for loading the audio file in chunks.
-   `umap-learn` for dimensionality reduction.
-   `hdbscan` for clustering.
-   `csv` for writing the results.

Here is the Python code to perform these steps:

python

`import wave
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

    mfcc_matrix = []

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
            mfcc_features_data = mfcc(audio_data[:, 0], samplerate=framerate, numcep=mfcc_features)

            # Average the MFCC across the chunk (to get a single MFCC vector per second)
            mfcc_avg = np.mean(mfcc_features_data, axis=0)
            mfcc_matrix.append(mfcc_avg)

            # Write the MFCC vector to the CSV file
            writer.writerow(mfcc_avg)

    wav.close()

    return np.array(mfcc_matrix)

# Step 2: Apply UMAP on the MFCC data
def apply_umap(mfcc_matrix, n_neighbors=15, min_dist=0.1, n_components=2):
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
def process_audio_and_cluster(wav_file, mfcc_output_csv="mfcc_output.csv", cluster_output_csv="cluster_output.csv"):
    # Process the audio file and compute MFCCs
    mfcc_matrix = process_wav_in_chunks(wav_file, output_csv=mfcc_output_csv)

    # Normalize MFCC data
    scaler = StandardScaler()
    mfcc_matrix_normalized = scaler.fit_transform(mfcc_matrix)

    # Apply UMAP
    umap_data = apply_umap(mfcc_matrix_normalized)

    # Apply HDBSCAN
    cluster_labels = apply_hdbscan(umap_data)

    # Save cluster results
    save_clusters_to_csv(cluster_labels, output_csv=cluster_output_csv)

# Example usage
process_audio_and_cluster('large_audio_file.wav')`

### Explanation:

1.  **process_wav_in_chunks**: This function reads the wave file in chunks (each chunk corresponds to 1 second of audio). It computes the MFCCs for each chunk and writes them to a CSV file.
2.  **apply_umap**: This function applies UMAP to reduce the dimensionality of the MFCC data.
3.  **apply_hdbscan**: This function applies HDBSCAN to cluster the UMAP-reduced data.
4.  **save_clusters_to_csv**: This function writes the clustering results to a CSV file.
5.  **process_audio_and_cluster**: This is the main function that orchestrates the entire process: reading the audio, computing MFCCs, applying UMAP and HDBSCAN, and saving the results.

You can adjust the UMAP and HDBSCAN parameters (`n_neighbors`, `min_dist`, `min_cluster_size`) based on the characteristics of your dataset.

##### You said:

give me code for argparse so we can specify the wavefile to process from the commandline

###### ChatGPT said:

ChatGPT

You can add argument parsing to the script using the `argparse` module to allow users to specify the wave file, MFCC output file, and cluster output file from the command line.

Here's how you can modify the script to include command-line arguments:

python

`import wave
import numpy as np
import csv
import argparse
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

    mfcc_matrix = []

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
            mfcc_features_data = mfcc(audio_data[:, 0], samplerate=framerate, numcep=mfcc_features)

            # Average the MFCC across the chunk (to get a single MFCC vector per second)
            mfcc_avg = np.mean(mfcc_features_data, axis=0)
            mfcc_matrix.append(mfcc_avg)

            # Write the MFCC vector to the CSV file
            writer.writerow(mfcc_avg)

    wav.close()

    return np.array(mfcc_matrix)

# Step 2: Apply UMAP on the MFCC data
def apply_umap(mfcc_matrix, n_neighbors=15, min_dist=0.1, n_components=2):
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
def process_audio_and_cluster(wav_file, mfcc_output_csv="mfcc_output.csv", cluster_output_csv="cluster_output.csv"):
    # Process the audio file and compute MFCCs
    mfcc_matrix = process_wav_in_chunks(wav_file, output_csv=mfcc_output_csv)

    # Normalize MFCC data
    scaler = StandardScaler()
    mfcc_matrix_normalized = scaler.fit_transform(mfcc_matrix)

    # Apply UMAP
    umap_data = apply_umap(mfcc_matrix_normalized)

    # Apply HDBSCAN
    cluster_labels = apply_hdbscan(umap_data)

    # Save cluster results
    save_clusters_to_csv(cluster_labels, output_csv=cluster_output_csv)

# Argument parsing function
def parse_args():
    parser = argparse.ArgumentParser(description="Process a wave file to extract MFCCs, apply UMAP, and cluster using HDBSCAN.")
    parser.add_argument('wav_file', type=str, help="Path to the wave file")
    parser.add_argument('--mfcc_output', type=str, default="mfcc_output.csv", help="Path to the output CSV file for MFCCs")
    parser.add_argument('--cluster_output', type=str, default="cluster_output.csv", help="Path to the output CSV file for cluster labels")

    return parser.parse_args()

# Main execution function
if __name__ == "__main__":
    args = parse_args()
    process_audio_and_cluster(args.wav_file, args.mfcc_output, args.cluster_output)`

### Command-line Usage

To run the script from the command line, you would use:

bash

`python script_name.py path_to_wave_file.wav --mfcc_output output_mfcc.csv --cluster_output output_cluster.csv`

-   **`path_to_wave_file.wav`**: Required argument. The path to the wave file that you want to process.
-   **`--mfcc_output`**: Optional argument. The path to the CSV file where the MFCC features per second will be stored (default is `mfcc_output.csv`).
-   **`--cluster_output`**: Optional argument. The path to the CSV file where the clustering results will be stored (default is `cluster_output.csv`).

By default, the MFCCs will be saved to `mfcc_output.csv` and the clusters to `cluster_output.csv`, but you can override this by specifying different filenames.

##### You said:

In python, read in a very long wave file that cannot fit in memory, and MFCC each second of the wavefile, write that matrix of MFCCs per second to a csv file, then compare that second against the past 10 seconds of MFCCs using cosine and euclidean distance, report those distances to a csv file

###### ChatGPT said:

ChatGPT

You can modify the code to compute the MFCCs per second for the wave file, store them, and then compute the cosine and Euclidean distances between each second and the previous 10 seconds. These distances can then be written to a CSV file.

Here's how to approach it:

1.  **Process the wave file in chunks**: For each second, compute the MFCCs and store them in memory.
2.  **Compare MFCCs of the current second to the last 10 seconds**: Calculate both the cosine and Euclidean distances between the current second's MFCCs and those from the last 10 seconds.
3.  **Write distances to CSV**: Save the distances (cosine and Euclidean) in a CSV file for each second.

### Key Python Libraries

-   `python_speech_features` for calculating MFCCs.
-   `scipy` for computing cosine and Euclidean distances.
-   `csv` for writing the output to CSV.

### Code Implementation:

python

`import wave
import numpy as np
import csv
from python_speech_features import mfcc
from scipy.spatial.distance import cosine, euclidean
import argparse

# Step 1: Read large wave file in chunks and compute MFCC
def process_wav_in_chunks(wav_file, chunk_duration_sec=1, mfcc_features=13, output_mfcc_csv="mfcc_output.csv", output_distance_csv="distances_output.csv"):
    # Open the wave file
    wav = wave.open(wav_file, 'r')
    framerate = wav.getframerate()
    num_channels = wav.getnchannels()
    bytes_per_sample = wav.getsampwidth()

    # Number of samples in one second
    chunk_size = framerate * chunk_duration_sec

    mfcc_matrix = []
    distances = []

    # Prepare the CSV writers
    with open(output_mfcc_csv, 'w', newline='') as mfcc_file, open(output_distance_csv, 'w', newline='') as distance_file:
        mfcc_writer = csv.writer(mfcc_file)
        distance_writer = csv.writer(distance_file)

        # Write the headers for the distances CSV
        distance_writer.writerow(['Second', 'Cosine_Distance', 'Euclidean_Distance'])

        second = 0
        while True:
            # Read the chunk of audio data
            raw_data = wav.readframes(chunk_size)

            if not raw_data:
                break

            # Convert raw audio data to numpy array
            audio_data = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, num_channels)

            # Compute MFCCs for the audio data (use only one channel if stereo)
            mfcc_features_data = mfcc(audio_data[:, 0], samplerate=framerate, numcep=mfcc_features)

            # Average the MFCC across the chunk (to get a single MFCC vector per second)
            mfcc_avg = np.mean(mfcc_features_data, axis=0)
            mfcc_matrix.append(mfcc_avg)

            # Write MFCC vector to the CSV
            mfcc_writer.writerow(mfcc_avg)

            # Compare against the previous 10 seconds of MFCCs if available
            if len(mfcc_matrix) > 10:
                mfcc_past_10_sec = mfcc_matrix[-11:-1]  # The last 10 seconds of MFCCs
                mfcc_current = mfcc_matrix[-1]  # Current second's MFCC

                # Calculate cosine and Euclidean distance between the current second and each of the last 10 seconds
                cosine_distances = [cosine(mfcc_current, past_mfcc) for past_mfcc in mfcc_past_10_sec]
                euclidean_distances = [euclidean(mfcc_current, past_mfcc) for past_mfcc in mfcc_past_10_sec]

                # Average these distances
                avg_cosine_distance = np.mean(cosine_distances)
                avg_euclidean_distance = np.mean(euclidean_distances)

                # Write the distances for this second to the CSV file
                distance_writer.writerow([second, avg_cosine_distance, avg_euclidean_distance])

            second += 1

    wav.close()

# Argument parsing function
def parse_args():
    parser = argparse.ArgumentParser(description="Process a wave file to extract MFCCs, compare with the past 10 seconds, and report distances.")
    parser.add_argument('wav_file', type=str, help="Path to the wave file")
    parser.add_argument('--mfcc_output', type=str, default="mfcc_output.csv", help="Path to the output CSV file for MFCCs")
    parser.add_argument('--distance_output', type=str, default="distances_output.csv", help="Path to the output CSV file for distance metrics")

    return parser.parse_args()

# Main execution function
if __name__ == "__main__":
    args = parse_args()
    process_wav_in_chunks(args.wav_file, output_mfcc_csv=args.mfcc_output, output_distance_csv=args.distance_output)`

### Explanation:

1.  **process_wav_in_chunks**:

    -   Reads the wave file in chunks, where each chunk corresponds to one second.
    -   Computes MFCCs for each second of audio.
    -   After each second, it compares the current MFCCs with the MFCCs from the previous 10 seconds using cosine and Euclidean distances.
    -   Stores the average cosine and Euclidean distances between the current second and the previous 10 seconds in a CSV file.
2.  **parse_args**:

    -   Uses the `argparse` module to allow the user to specify the input wave file, the output CSV file for MFCCs, and the output CSV file for distances from the command line.

### Command-Line Usage:

To run the script from the command line, you would use:

bash

`python script_name.py path_to_wave_file.wav --mfcc_output output_mfcc.csv --distance_output output_distances.csv`

-   **`path_to_wave_file.wav`**: Required argument. The path to the wave file that you want to process.
-   **`--mfcc_output`**: Optional argument. The path to the CSV file where the MFCC features per second will be stored (default is `mfcc_output.csv`).
-   **`--distance_output`**: Optional argument. The path to the CSV file where the distances (cosine and Euclidean) between the current second and the past 10 seconds will be stored (default is `distances_output.csv`).