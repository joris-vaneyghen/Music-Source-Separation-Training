import csv
import os
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

valid_path = "dataset/jazz-dataset-mss/valid"
valid_path_dest = "dataset/valid"

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
        piano, _ = sf.read(os.path.join(subdir_path, "piano.wav"))

        # Ensure all arrays have the same shape by taking the minimum length
        min_length = min(drums.shape[0], piano.shape[0])
        drums = drums[:min_length]
        piano = piano[:min_length]

        # Create silent stereo audio with same length
        silence = np.zeros((min_length, 2))

        # Overlay the tracks by summing
        mixture = drums + piano

        # Save the mixture and silence for other and bass in destination directory
        sf.write(os.path.join(subdir_dest_path, "mixture.wav"), mixture, sr)
        sf.write(os.path.join(subdir_dest_path, "other.wav"), silence, sr)
        sf.write(os.path.join(subdir_dest_path, "bass.wav"), silence, sr)

        # Also copy drums.wav and piano.wav to destination (though they should already be copied above)
        shutil.copy2(os.path.join(subdir_path, "drums.wav"), subdir_dest_path)
        shutil.copy2(os.path.join(subdir_path, "piano.wav"), subdir_dest_path)


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


# Create silent stereo audio
silence = np.zeros((44100 * 100, 2))
os.makedirs("dataset/silence", exist_ok=True)
sf.write("dataset/silence/other.wav", silence, 44100)
sf.write("dataset/silence/bass.wav", silence, 44100)

directories_to_search = [
    ('dataset/jazz-dataset-mss/train',['drums','piano']),
    ('dataset/jazz-extra-dataset-mss',['drums','piano']),
    ('dataset/jazz-extra-2-dataset-mss',['drums','piano']),
    ('dataset/silence',['bass','other']),
]
output_csv = 'dataset/train/dataset.csv'
os.makedirs("dataset/train", exist_ok=True)
write_instrument_csv(directories_to_search, output_csv)
print(f"CSV file created at: {output_csv}")