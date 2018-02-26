# Handwriting Synthesis
- [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/4QuantOSS/handwriting-synthesis/master) [![Binder]( https://img.shields.io/badge/launch-jupyterlab-red.svg)](https://mybinder.org/v2/gh/4QuantOSS/handwriting-synthesis/master?urlpath=lab)

Implementation of the handwriting synthesis experiments in the paper <a href="https://arxiv.org/abs/1308.0850">Generating Sequences with Recurrent Neural Networks</a> by Alex Graves.  The implementation very closely follows the original paper, with a few slight deviations, and the generated samples are of similar quality to those presented in the paper.

Below are a few hundred samples from the model, including some samples demonstrating the effect of priming and biasing the model.  Loosely speaking, biasing controls the neatness of the samples and priming controls the style of the samples. The code for these demonstrations can be found in `demo.py` and should be fairly easy to modify for your own purposes.  A pretrained model is also included in `checkpoints/`.

## Notebooks

- [![Binder](https://img.shields.io/badge/launch-demo%20notebook-green.svg)](https://mybinder.org/v2/gh/4QuantOSS/handwriting-synthesis/master?filepath=notebooks%2Fdemo.ipynb)

## Demo #1
The following samples were generated with a fixed style and fixed bias.

**Smash Mouth – All Star (<a href="https://www.azlyrics.com/lyrics/smashmouth/allstar.html">lyrics</a>)**
![](img/all_star.svg)

## Demo #2
The following samples were generated with varying style and fixed bias.  Each verse is generated in a different style.

**Vanessa Carlton – A Thousand Miles (<a href="https://www.azlyrics.com/lyrics/vanessacarlton/athousandmiles.html">lyrics</a>)**
![](img/downtown.svg)

## Demo #3
The following samples were generated with a fixed style and varying bias.  Each verse has a lower bias than the previous, with the last verse being unbiased.

**Leonard Cohen – Hallelujah (<a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ">lyrics</a>)
![](img/give_up.svg)**
