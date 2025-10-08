# A Systematic Review of Heart Sound Detection Algorithms: Experimental Results and Insights

## Contents
- [Introduction](#introduction)  
- [Methods](#methods)  
- [Preprocessing](#preprocessing)  
- [Segmentation](#segmentation)  
- [Feature extraction](#feature-extraction)  
- [Classification](#classification)  
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
