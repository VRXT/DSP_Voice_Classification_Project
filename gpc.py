import os
import csv
import numpy as np
import librosa
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# =========================
# Gender Classification Script
# =========================
# This script iterates over all MP3 files in data/audio/, classifies the speaker's gender using pitch analysis,
# compares the results with ground truth from data/meta.csv, and outputs detailed statistics and a results CSV.
# The code is heavily commented for educational purposes.

# Output CSV file
RESULTS_CSV = "results.csv"
# Path to metadata CSV (ground truth)
META_CSV = os.path.join("data", "meta.csv")
# Directory containing audio samples
AUDIO_DIR = os.path.join("data", "audio")

# --- Algorithm: Gender Classification by Pitch ---
# We use librosa to extract the fundamental frequency (f0) from each audio file.
# - If the mean f0 is between 85–150 Hz, we classify as 'male'.
# - If the mean f0 is between 150–255 Hz, we classify as 'female'.
# - Otherwise, or if pitch cannot be determined, 'unclassified'.
def classify_gender(audio_path):
    """
    Classifies the gender of a speaker in an audio file based on pitch (fundamental frequency).
    Returns a tuple: (predicted_label, mean_frequency)
    """
    try:
        # Load audio file
        audio_data, fs = librosa.load(audio_path, sr=None)
        # Extract pitch using librosa's pyin
        f0, _, _ = librosa.pyin(audio_data, fmin=50, fmax=300, sr=fs)
        # Remove NaN values (unvoiced frames)
        f0 = f0[~np.isnan(f0)]
        if len(f0) == 0:
            return 'unclassified', None
        mean_freq = np.mean(f0)
        # Apply pitch-based gender rules
        if 85 <= mean_freq <= 150:
            return 'male', mean_freq
        elif 150 < mean_freq <= 255:
            return 'female', mean_freq
        else:
            return 'unclassified', mean_freq
    except Exception as e:
        print(f"[ERROR] Error processing {audio_path}: {e}")
        return 'error', None

# --- Load ground truth from meta.csv ---
def load_ground_truth(meta_csv):
    """
    Loads the ground truth gender for each audio file from meta.csv.
    Returns a dict: filename -> gender (male/female/other)
    """
    truth = {}
    with open(meta_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize filename and gender
            fname = row['filename'].strip()
            gender = row.get('gender', '').strip().lower()
            truth[fname] = gender
    return truth

def process_file(args):
    fname, ground_truth = args
    audio_path = os.path.join(AUDIO_DIR, fname)
    pred_label, mean_freq = classify_gender(audio_path)
    true_label = ground_truth.get(fname, 'unknown')
    # Only count as correct if prediction matches ground truth and is male/female
    correct = pred_label == true_label and pred_label in ['male', 'female']
    unclassified = pred_label not in ['male', 'female']
    incorrect = (pred_label in ['male', 'female']) and (pred_label != true_label)
    return {
        'file': fname,
        'predicted': pred_label,
        'mean_freq': f"{mean_freq:.2f}" if mean_freq is not None else '',
        'ground_truth': true_label,
        'correct': correct,
        'unclassified': unclassified,
        'incorrect': incorrect
    }

# --- Main analysis and comparison ---
def main():
    parser = argparse.ArgumentParser(description="Gender classification from audio using pitch analysis.")
    parser.add_argument('-a', '--all', action='store_true', help='Process all audio files (default: only first 100)')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads for parallel processing (default: 1, safe for macOS)')
    args = parser.parse_args()

    start_time = time.time()
    ground_truth = load_ground_truth(META_CSV)
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.mp3')]
    audio_files.sort()  # For reproducibility
    if not args.all:
        audio_files = audio_files[:10]
    results = []
    correct = 0
    incorrect = 0
    unclassified = 0
    total = 0
    if args.threads > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            tasks = [(fname, ground_truth) for fname in audio_files]
            futures = [executor.submit(process_file, t) for t in tasks]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Classifying audio files"):
                row = f.result()
                results.append(row)
                if row['correct']:
                    correct += 1
                elif row['unclassified']:
                    unclassified += 1
                elif row['incorrect']:
                    incorrect += 1
                total += 1
    else:
        for fname in tqdm(audio_files, desc="Classifying audio files"):
            row = process_file((fname, ground_truth))
            results.append(row)
            if row['correct']:
                correct += 1
            elif row['unclassified']:
                unclassified += 1
            elif row['incorrect']:
                incorrect += 1
            total += 1
    # Write results to CSV
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file', 'predicted', 'mean_freq', 'ground_truth', 'correct']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})
    # Print statistics
    print("\n--- Classification Statistics ---")
    print(f"Total files: {total}")
    print(f"Correct (male/female): {correct}")
    print(f"Incorrect (male/female): {incorrect}")
    print(f"Unclassified: {unclassified}")
    if total > 0:
        print(f"Success rate: {correct/total:.2%}")
        print(f"Failure rate: {incorrect/total:.2%}")
        print(f"Unclassified rate: {unclassified/total:.2%}")
    print(f"Results written to {RESULTS_CSV}")
    elapsed = time.time() - start_time
    print(f"Total time elapsed: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
