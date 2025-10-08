# A Systematic Review of Heart Sound Detection Algorithms: Experimental Results and Insights

## Introduction
- Problem: automatic heart-sound detection helps early CVD diagnosis.  
- Goal: compare algorithms and features to say which work best where.  
- Outcome: give practical guidance and point out problems in prior work.

## Methods (processing pipeline)
- Pipeline steps: preprocessing → segmentation → feature extraction → classification.  
- Comparison approach: evaluate techniques on common datasets with the same comparison level.  
- Focus: segmentation settings and nine key heart-sound features.

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
