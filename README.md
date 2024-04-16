# Audio processing with neural controlled differential equations

Neural controllled differential equations (nCDEs) work well for time series - I'm exploring their utility for audio processing

## Audio source separation (demixing)

As a first experiment, I am using gated neural differential equations (https://arxiv.org/pdf/2307.06398.pdf) to try and perform audio source separation ont he musdb18 dataset (https://sigsep.github.io/datasets/musdb.html)


## Requirements

This project makes use of the JAX library and related ecosystem of libraries (optax, equionox, diffrax)
