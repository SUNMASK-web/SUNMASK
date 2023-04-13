# Overview
We release different variations of the SUNMASK concept for learning and demonstration, though the current code release is not out of the box usable it should be useful to build on for future work.

# Demo / Learning About SUNMASK (Preferred)
`demo` contains a simplified variation of SUNMASK, using a PerceiverAR style core model alongside the masking and denoising objective. This should be generally usable and useful, although not directly comparable to Coconet or the published work in SUNMASK due to the changes in the dataset processing and context conditioning of the model.

This demo variant uses a subset of the overall JSB data, processed using the following file + branch of my experimental library, which adds per-file attribution as well as meter information about each piece.
[https://github.com/kastnerkyle/kkpthlib/blob/basics/examples/scripts/make_simple_music_dataset.py](https://github.com/kastnerkyle/kkpthlib/blob/basics/examples/scripts/make_simple_music_dataset.py)

I recommend simply using this cleaned data as-is for now in the demo code.

# Raw Experiment Dump
`raw_experimental` contains the code from our colab notebooks used for experiments in the paper, along with various supplemental materials. These should be runnable by copying the code back into colab form, but we keep the raw information separate from the notebook format.

The raw experimental code uses the full JSB dataset contained here [https://github.com/czhuang/JSB-Chorales-dataset](https://github.com/czhuang/JSB-Chorales-dataset), as well as the respective text datasets `text8` and `emnlp2017`.
