# A Systematic Review of Heart Sound Detection Algorithms: Experimental Results and Insights

## Contents
- [Introduction](#introduction)  
- [Methods](#methods)  
- [Preprocessing](#preprocessing)  
- [Segmentation](#segmentation)  
- [Feature extraction](#feature-extraction)  
- [Classification](#classification)  
- [Datasets](#datasets)  
- [Evaluation metrics](#evaluation-metrics)  
- [Experiments and results](#experiments-and-results)  
- [Discussion and insights](#discussion-and-insights)  
- [Conclusion and recommendations](#conclusion-and-recommendations)  
- [Nine salient heart-sound features (short definitions)](#nine-salient-heart-sound-features-short-definitions)

## Introduction
- Problem: automatic heart-sound detection helps early CVD diagnosis.  
- Goal: compare algorithms and features to say which work best where.  
- Outcome: give practical guidance and point out problems in prior work.

## Methods
- Broad scope: describes datasets, evaluation protocol, and the overall experimental design.  
- Central idea: the authors structure the method around a processing pipeline (preprocessing → segmentation → feature extraction → classification).  
- Comparison approach: evaluate techniques on common datasets with the same comparison level.

## Preprocessing
- Purpose: clean and standardize raw PCG signals so later steps (segmentation, features) work reliably.  
- Common tasks:
  - Filtering: bandpass filters to remove out-of-band noise (e.g., 20–400 Hz typical for heart sounds).  
  - Denoising: wavelet denoising or spectral subtraction for background noise reduction.  
  - Normalization / amplitude scaling: make signal amplitudes comparable across recordings.  
  - Resampling: unify sampling rates across datasets (e.g., 2 kHz or 4 kHz) to use the same feature parameters.  
  - Artifact removal: detect and remove movement, stethoscope clicks, or very noisy segments.  
  - Baseline wander removal: high-pass filtering to remove slow drifts.
- Why it matters: poor preprocessing changes feature values and can degrade segmentation and classifier performance.

## Segmentation
- Window length: include at least one complete cardiac period (S1–S2 cycle).  
- Overlap: use a large overlap rate to avoid missing transients and boundaries.  
- Effect: improves robustness and reduces sensitivity to window placement.

## Feature extraction
- TIME & FFT (single-variable): good with traditional classifiers and deep models.  
- STFT, CWT, S-transform (2D time–frequency): robust across classifier types.  
- MFCC / Mel-spectrum: works best when paired with deep neural networks.

## Classification
- Traditional classifiers (SVM, RF, etc.): effective with strong handcrafted features.  
- Deep neural networks: exploit richer features (especially MFCC, TIME) for best results.  
- Recommendation: match feature type to classifier capability.

## Datasets
Below are the datasets used in the paper’s experiments and the dataset table summary (Table III in the paper). The experiments specifically use three datasets: PhysioNet/CinC Challenge Dataset (PCCD), Pediatric Heart Sound Dataset (PHSD), and PASCAL Heart Sound Classification Challenge Dataset (PHSCCD). The table reproduced here summarizes the public datasets listed in the paper.

| Dataset (Used in Paper) | # of files | # of Classes | SF (Hz) | Balance | Time (s) | Collection Method / Device | States (labels reported) |
|---|---:|---:|---:|---:|---:|---|---|
| WHSM (No) | 16 | Multiple | NR | NR | 9 | Stethoscope | S1 \ Sys \ S2 \ Dia \ S3 \ S4 |
| MHSML (No) | 23 | Multiple | 44100 | N | 7–9 | Not Reported (NR) | S1 \ Sys \ S2 \ Dia |
| CAHM (No) | 64 | Multiple | NR | NR | NR | NR | NR |
| PHSCCD (A,B) (Yes) | 176 (A) / 656 (B) | 4 (A) / 3 (B) | Below 195 | N | 1–30 | iStethoscope Pro (iPhone app) | S1 \ Sys \ S2 \ Dia |
| PHSD (Yes) | 528 | Multiple | 44100 | N | 3–249 | Digital stethoscope (ThinklabsOne) | S1 \ Sys \ S2 \ Dia \ S3 \ S4 |
| PCCD (PhysioNet/CinC) (Yes) | 3240 | 2 | 2000 (resampled) | N | 5–120 | Multiple-database fusion; electronic devices | S1 \ Sys \ S2 \ Dia |

Notes and key dataset details used in experiments:
- PCCD (PhysioNet/CinC Challenge Dataset): largest dataset used in experiments — 3,240 recordings collected from multiple research groups; recordings resampled from 44.1 kHz to 2,000 Hz for experiments; distribution: 2,575 normal, 665 abnormal (realistic imbalance). PCG durations 5–120 s.
- PHSD (Pediatric Heart Sound Dataset): pediatric recordings — 528 files (≈4 hours total), durations 3–249 s, recorded from children aged 1 month–12 years using ThinklabsOne at 44.1 kHz / 16-bit; multi-class labels (normal, ASD, VSD, TOF, other).
- PHSCCD (PASCAL Heart Sound Classification Challenge Dataset): two subdatabases (A and B). Database A: 176 files (quadruple classification). Database B: 656 files (triple classification). File lengths 1–30 s. The authors also used a small two-class subset (normal vs murmur) for some experiments.

## Evaluation metrics
The paper defines and uses the following core evaluation metrics for binary heart-sound detection (normal vs abnormal). The formulas below are presented exactly as in the paper.

Let TP = true positives, TN = true negatives, FP = false positives, FN = false negatives. In this article, TP denotes correctly detected abnormal heart sounds.

- Accuracy (Acc)  
  Acc = (TP + TN) / (TP + TN + FP + FN)

- Specificity (Spe)  
  Spe = TN / (TN + FP)

- Sensitivity / Recall (Sen)  
  Sen = TP / (TP + FN)

- Precision (Pre)  
  Pre = TP / (TP + FP)

- F1-score (F1)  
  F1 = 2 × Pre × Sen / (Pre + Sen)

Note from paper: the heart-sound datasets are generally imbalanced, so Accuracy is not a suitable sole metric — the paper emphasizes F1-score as a critical evaluation metric for comparing methods on imbalanced datasets.

## Experiments and results
- Setup: same-level comparisons across public heart-sound datasets.  
- Key findings: segmentation with one period + high overlap; STFT/CWT/ST robust; TIME/FFT strong for many setups; MFCC shines with DNNs.  
- Best combos observed: TIME + DNN or MFCC + DNN.

## Discussion and insights
- Main issues: inconsistent preprocessing/segmentation across studies and lack of standard benchmarks.  
- Practical insight: choose segmentation and overlap carefully and align features to model type.  
- Suggestion: more standardized datasets and clearer reporting needed.

## Conclusion and recommendations
- Clear rules: use windows containing ≥1 heart cycle, high overlap; pair features and models intentionally.  
- Best-practice combos: TIME or MFCC with DNNs gave top performance in experiments.  
- Future work: standard benchmarks, explore hybrid features and richer deep models.

## Nine salient heart-sound features (short definitions)
1. Time-domain features (TIME)  
   - Simple measures computed directly from the signal waveform (e.g., peak amplitude, signal energy, root-mean-square, zero-crossing rate). Good for capturing overall loudness and temporal patterns.

2. FFT-based spectral features (FFT)  
   - Frequency-domain descriptors derived from the (global) Fourier transform such as dominant frequency, spectral energy in bands, and spectral peaks; useful for identifying frequency content of heart sounds.

3. Short-Time Fourier Transform (STFT) / Spectrogram  
   - A time–frequency representation that computes FFTs over short, overlapping windows to produce a spectrogram; captures how spectral content evolves over time.

4. Continuous Wavelet Transform (CWT) / Scalogram  
   - A multi-scale time–frequency analysis using wavelets; provides good localization of transient events at different scales and is effective for nonstationary heart sounds.

5. S-Transform (ST)  
   - A hybrid time–frequency transform combining aspects of STFT and wavelets; produces a time–frequency map with frequency-dependent resolution useful for transient detection.

6. Mel-spectrum and Mel-Frequency Cepstral Coefficients (MFCC)  
   - Perceptually motivated spectral features: mel-spectrum maps frequency to the Mel scale; MFCCs are compact cepstral coefficients extracted from the mel-spectrum, often effective with neural networks.

7. Wavelet coefficient statistics / Wavelet-packet features  
   - Features derived from (discrete) wavelet decomposition—e.g., statistics (mean, variance, entropy) of coefficients across bands—capturing multi-resolution signal properties.

8. Spectral-shape measures (centroid, bandwidth, roll-off, flux)  
   - Aggregate spectral descriptors that summarize spectral “shape” and movement: spectral centroid (center of mass), bandwidth (spread), roll-off (cutoff frequency), and flux (frame-to-frame change).

9. Statistical / information-theoretic features (entropy, skewness, kurtosis, SNR)  
   - Higher-order descriptive features capturing signal complexity and distribution (e.g., Shannon entropy, skewness, kurtosis) and noise/quality metrics like signal-to-noise ratio; useful for characterizing abnormal vs. normal sounds.
