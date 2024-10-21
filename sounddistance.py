# Copyright (c) 2024 Abram Hindle
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
import wave
import numpy as np
import csv
from python_speech_features import mfcc
from scipy.spatial.distance import cosine, euclidean
import argparse

# Step 1: Read large wave file in chunks and compute MFCC
def process_wav_in_chunks(wav_file, chunk_duration_sec=1, mfcc_features=13, output_mfcc_csv="mfcc_output.csv", output_distance_csv="distances_output.csv", warn=2, last_n_seconds=10):
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
        distance_writer.writerow(['Second', 'Max_Cosine_Distance', 'Max_Euclidean_Distance', 'Cosine_Distance', 'Euclidean_Distance'])
        
        second = 0
        last_euclidean_average = None
        while True:
            # Read the chunk of audio data
            raw_data = wav.readframes(chunk_size)
            
            if not raw_data:
                break
            
            # Convert raw audio data to numpy array
            audio_data = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, num_channels)
            
            # Compute MFCCs for the audio data (use only one channel if stereo)
            mfcc_features_data = mfcc(audio_data[:, 0], nfft=2048, samplerate=framerate, numcep=mfcc_features)
            
            # Average the MFCC across the chunk (to get a single MFCC vector per second)
            mfcc_avg = np.mean(mfcc_features_data, axis=0)
            mfcc_matrix.append(mfcc_avg)
            
            # Write MFCC vector to the CSV
            mfcc_writer.writerow(mfcc_avg)
            # Compare against the previous 10 seconds of MFCCs if available
            if len(mfcc_matrix) > last_n_seconds:
                mfcc_past_10_sec = mfcc_matrix[-(last_n_seconds+1):-1]  # The last 10 seconds of MFCCs
                mfcc_current = mfcc_matrix[-1]  # Current second's MFCC

                # Calculate cosine and Euclidean distance between the current second and each of the last 10 seconds
                cosine_distances = [cosine(mfcc_current, past_mfcc) for past_mfcc in mfcc_past_10_sec]
                euclidean_distances = [euclidean(mfcc_current, past_mfcc) for past_mfcc in mfcc_past_10_sec]

                # Average these distances
                avg_cosine_distance = np.mean(cosine_distances)
                avg_euclidean_distance = np.mean(euclidean_distances)
                max_cosine_distance = np.max(cosine_distances)
                max_euclidean_distance = np.max(euclidean_distances)
                if not (last_euclidean_average is None):
                    if avg_euclidean_distance >= warn*last_euclidean_average:
                        print(f'In second {second} a large euclidean distance was observed {avg_euclidean_distance} versus {last_euclidean_average}')
                last_euclidean_average = avg_euclidean_distance
                # Write the distances for this second to the CSV file
                distance_writer.writerow([second, max_cosine_distance, max_euclidean_distance ,avg_cosine_distance, avg_euclidean_distance])
            
            second += 1

    wav.close()

# Argument parsing function
def parse_args():
    parser = argparse.ArgumentParser(description="Process a wave file to extract MFCCs, compare with the past 10 seconds, and report distances.")
    parser.add_argument('wav_file', type=str, help="Path to the wave file")
    parser.add_argument('--mfcc_output', type=str, default="mfcc_output.csv", help="Path to the output CSV file for MFCCs")
    parser.add_argument('--distance_output', type=str, default="distances_output.csv", help="Path to the output CSV file for distance metrics")
    parser.add_argument('--warn',type=float,default=2, help="How many times the prior average before you are warned of a change")
    parser.add_argument('--seconds',type=float,default=10, help="How many prior seconds")
    return parser.parse_args()

# Main execution function
if __name__ == "__main__":
    args = parse_args()
    process_wav_in_chunks(args.wav_file, output_mfcc_csv=args.mfcc_output, output_distance_csv=args.distance_output, warn=args.warn, last_n_seconds=args.seconds)
