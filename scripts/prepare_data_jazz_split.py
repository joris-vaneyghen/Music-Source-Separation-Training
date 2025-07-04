import csv
import os
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

valid_path = "dataset/jazz-dataset-mss/valid"
valid_path_dest = "dataset/valid_split"

# Create destination directory if it doesn't exist
os.makedirs(valid_path_dest, exist_ok=True)

for subdir in os.listdir(valid_path):
    subdir_path = os.path.join(valid_path, subdir)
    subdir_dest_path = os.path.join(valid_path_dest, subdir)

    if os.path.isdir(subdir_path):
        # Create destination subdirectory
        os.makedirs(subdir_dest_path, exist_ok=True)

        # Read the audio files
        drums, sr = sf.read(os.path.join(subdir_path, "drums.wav"))
        bass, _ = sf.read(os.path.join(subdir_path, "bass.wav"))
        piano, _ = sf.read(os.path.join(subdir_path, "piano.wav"))

        # Ensure all arrays have the same shape by taking the minimum length
        min_length = min(drums.shape[0], bass.shape[0], piano.shape[0])
        drums = drums[:min_length]
        bass = bass[:min_length]
        piano = piano[:min_length]

        # Convert to mono if they're stereo
        if drums.ndim > 1:
            drums_L = drums[:min_length, 0]
            drums_R = drums[:min_length, 1]
        else:
            drums_L = drums[:min_length]
            drums_R = drums[:min_length]

        if bass.ndim > 1:
            bass = bass[:min_length, 0]  # Take left channel for mono
        else:
            bass = bass[:min_length]

        if piano.ndim > 1:
            piano = piano[:min_length, 1]  # Take right channel for mono
        else:
            piano = piano[:min_length]

        # Create hard-panned stereo versions for the mixture
        bass_stereo = np.column_stack([bass, np.zeros_like(bass)])  # Hard panned left
        piano_stereo = np.column_stack([np.zeros_like(piano), piano])  # Hard panned right

        # Create the stereo mixture
        mixture = np.zeros((min_length, 2))
        mixture[:, 0] += drums_L if drums.ndim == 1 else drums[:, 0]
        mixture[:, 1] += drums_R if drums.ndim == 1 else drums[:, 1]
        mixture += bass_stereo
        mixture += piano_stereo

        # Save the mixture and silence for other and piano in destination directory
        sf.write(os.path.join(subdir_dest_path, "mixture.wav"), mixture, sr)

        sf.write(os.path.join(subdir_dest_path, "drums_L.wav"), drums_L, sr)
        sf.write(os.path.join(subdir_dest_path, "drums_R.wav"), drums_R, sr)
        sf.write(os.path.join(subdir_dest_path, "bass.wav"), bass, sr)
        sf.write(os.path.join(subdir_dest_path, "piano.wav"), piano, sr)


def find_wav_files(dirs):
    """Recursively find all .wav files in the given directories and yield (instrument, path) pairs."""
    for (dir_path, instruments) in dirs:
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith('.wav'):
                    # Extract instrument name (assumes format is "instrument[...].wav")
                    instrument = file.split('_')[0].split('.')[0].lower()
                    if instrument in instruments:
                        full_path = Path(root) / file
                        yield instrument, str(full_path)


def write_instrument_csv(dirs, output_file):
    """Write a CSV file listing all instruments and their paths."""
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['instrum', 'path'])

        for instrument, path in find_wav_files(dirs):
            writer.writerow([instrument, path])


directories_to_search = [
    ('dataset/jazz-dataset-mss/train',['drums','bass','piano']),
    ('dataset/jazz-extra-dataset-mss',['drums','bass','piano']),
    ('dataset/jazz-extra-2-dataset-mss',['drums','bass','piano']),
]
output_csv = 'dataset/train_split/dataset.csv'
os.makedirs("dataset/train_split", exist_ok=True)
write_instrument_csv(directories_to_search, output_csv)
print(f"CSV file created at: {output_csv}")