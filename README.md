# Multimodal Neural Style Transfer for Face-to-Sketch Synthesis (CUFS)

Final-year project report repository  CM3015 Machine Learning and Neural Networks,
Project template 3.1 (Project Idea 1: Neural Style Transfer).

## Overview

A systematic study of classical, optimisation-based neural style transfer (Gatys et al., 2016)
applied to face-photograph-to-pencil-sketch translation on the CUHK Face Sketch Database (CUFS).
The project covers: loss-landscape calibration and a controlled hyperparameter study; a
multimodal extension that blends Gram-matrix style statistics from multiple exemplar sketches
(two mathematically distinct blending strategies); a cross-subject identity-preservation study
under four generation conditions evaluated on a held-out 38-subject test partition; content-style
recombination and style interpolation; a three-architecture comparison (VGG16, VGG19, ResNet50);
and a comparison against a pretrained feed-forward arbitrary-style-transfer network.

**Central finding:** facial identity, measured by face-embedding cosine similarity, survives
cross-subject style transfer at a statistically significant level even when the target subject's
own sketch is entirely withheld from generation. A recurring secondary finding is that standard
image-similarity metrics (SSIM, FSIM, LPIPS) repeatedly fail to detect differences that an
identity-embedding metric catches.

## Repository structure

```javascript
project notebook (Google Colab, single NVIDIA T4 GPU)
figures/     result figures used in the report
```

## Running the project

The notebook is designed for Google Colab (runtime: GPU). Open notebook in Colab and run
top to bottom. Reproducing the full results takes roughly two to three hours of GPU time (the dominant cost is 250-iteration L-BFGS optimisation for each generated image).

### Requirements

See `requirements.txt`. Versions reflect the Colab environment at the time of the experiments;
PyTorch is used for the optimisation-based pipeline and TensorFlow/TensorFlow Hub for the
pretrained feed-forward stylisation network in the minimal-model comparison.

## Notes on reproduction

- The train/test partition (150 subjects development / 38 held out) is fixed by an explicit,
hard-coded subject list in the notebook and is identical in every run.
- In cross-subject conditions, partner ("distractor") sketches are assigned by randomised
pairing rather than a fixed hand-chosen set. This is deliberate: it prevents results from
depending on any particular pairing, and functions as a lightweight robustness check. All
headline statistical conclusions (own-sketch similarity significantly exceeding
other-sketch similarity, p < 0.0001, n = 38) replicated across independent sessions.
- Rerunning the notebook end-to-end therefore regenerates all results; individual metric
values may differ in the last decimals while all reported conclusions hold.


## References

Gatys, L.A., Ecker, A.S. and Bethge, M. (2016) 'Image Style Transfer Using Convolutional
Neural Networks', CVPR. | Wang, X. and Tang, X. (2009) 'Face Photo-Sketch Synthesis and
Recognition', IEEE TPAMI 31(11). | Full reference list in the report.
