# DSP Project: Gender Classification from Audio using Pitch

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle&logoColor=white)
![Mozilla](https://img.shields.io/badge/Mozilla-Common%20Voice-orange?logo=mozilla&logoColor=white)

## Acknowledgment

This project was completed by **Mushari T. Alshqar** (_Student ID: 201501201_) and **Ali Al-Yasin** (_Student ID: 201901784_) under the supervision of **Dr. Abul Bashar**. It was submitted on **May 5th, 2025**, as part of the final project for the _Digital Signal Processing (COEN 4322)_ course at **Prince Mohammed Bin Fahd University (PMU)**, within the _College of Computer Engineering and Sciences (CCES)_.

| **Contributor**      | **Student ID** | **Role**            |
| -------------------- | -------------- | ------------------- |
| _Mushari T. Alshqar_ | 201501201      | Project Contributor |
| _Ali Al-Yasin_       | 201901784      | Project Contributor |
| **Dr. Abul Bashar**  | -              | Supervisor          |

## Overview

This project analyzes audio files in `data/audio/`, classifies the speaker's gender (male/female/unclassified) based on pitch (fundamental frequency), and compares the results to ground truth labels in `data/meta.csv`. The results and detailed statistics are written to `results.csv`.

## Features

- **Pitch-based gender classification**: Uses `librosa` to estimate pitch and classify gender.
- **Progress bars**: See real-time progress for audio processing.
- **CSV results**: Logs each file's classification, ground truth, and summary statistics.
- **No dataset download**: The script works with your local audio and metadata files only.

## Requirements

- Python 3.8+
- Packages: `librosa`, `numpy`, `tqdm`

## Setup

1. **Install dependencies:**
   ```sh
   pip install librosa numpy tqdm
   ```
2. **Prepare your data:**
   - Place your audio files in `data/audio/`.
   - Ensure `data/meta.csv` contains the ground truth labels for each audio file.

## Usage

Run the script (by default, only the first 10 audio files are processed):

```sh
python gpc.py
```

To process all audio files, use the `-a` or `--all` flag:

```sh
python gpc.py -a
```

You can also control the number of threads (for parallel processing) with the `--threads` option:

```sh
python gpc.py --threads 2
```

- The script uses multithreading to speed up audio analysis if you specify more than one thread.
- Progress bars will show classification status.

## Output

- `results.csv`: Contains file name, predicted label, mean pitch frequency, ground truth, and correctness.
- CLI output: Shows total files processed, success/failure/unclassified counts, and rates.

## How it works

- **Pitch extraction:** Uses `librosa.pyin` to estimate the fundamental frequency (f0) for each audio file.
- **Classification:**
  - Male: 85–150 Hz
  - Female: 150–255 Hz
  - Unclassified: Outside these ranges or if pitch cannot be determined
- **Comparison:**
  - The script compares its prediction to the ground truth in `meta.csv` and computes statistics.

## Example CLI Output

```
Classifying audio files: 100%|██████████| 100/100 [00:10<00:00, 10.00it/s]

--- Classification Statistics ---
Total files: 100
Correct (male/female): 80
Incorrect (male/female): 15
Unclassified: 5
Success rate: 80.00%
Failure rate: 15.00%
Unclassified rate: 5.00%
Results written to results.csv
```

## License

This project uses the [Common Voice dataset](https://www.kaggle.com/datasets/mozillaorg/common-voice) under the CC0-1.0 license.

---

_Created with ❤️ by Mushari and Ali._
