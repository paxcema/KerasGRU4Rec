# GRU4Rec in Keras

This repository offers an implementation of the "Session-based Recommendations With Recurrent Neural Networks" paper using the Keras framework, tested with TensorFlow backend.

Included are some tests that use the MovieLens 20M dataset. Our implementation presents comparable results to that of the original Theano implementation offered by the GRU4Rec authors for this dataset, and also one of the original ones: 2015 RecSys Challenge dataset (RSC15).

Aditionally, a script can be found for determining a dataset's Dwell Time information, as seen on "Incorporating Dwell Time in Session-Based Recommendations with Recurrent Neural Networks" (http://ceur-ws.org/Vol-1922/paper11.pdf). Used with the RSC15 dataset, augmentation results can be reproduced, although we have not been able to replicate the final reported performance metrics.

Credit goes to yhs-968 for his parallel batcher, as shown in his pyGRU4REC repository.

To run the RNN model, refer to the Scratch/M4-WithDwellTime.ipynb notebook.

To run the Dwell Time augmentation process, refer to the DwellTimeTheano/DwellTimeTheano.ipynb notebook.

A repository clean-up is on the shortlist, try to ignore the disorder in the meantime.

Future work contemplates incorporating dwell time in an online manner to the model, hoping to leverage said information in the learning process, instead of in a previous preprocessing stage.

## Original GRU4Rec Readme

Basic implementation of the algorithm of "Session-based Recommendations With Recurrent Neural Networks" (http://arxiv.org/abs/1511.06939). With the extensions introduced in "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations" (http://arxiv.org/abs/1706.03847).

Please always use the latest version as baseline and cite both papers when you do so! Parameters for the Recsys Challenge 2015 (RSC15) dataset can be found in examples/rsc15/run_rsc15.py

The code was optimized for GPU execution, running it on CPU is not optimal.

**LICENSING:** proper licence text coming soon! Meanwhile these are the main guidelines: for research and education purposes the code is and always will be free to use. Use in commercial systems of the code or parts of it requires a licence. If you've been using the code or any of its derivates in a commercial system, contact me.

## Requirements

I advise to use latest Theano version (Theano-1.0.0 or newer). The code was tested with various numpy and pandas versions under python 3.4 and 3.6. Python 2 is not supported.

## Update 08-06-2018
- Refactor and cleaning.
- Speeding up execution.
- Ease of life improvements.
- Code for evaluating on GPU.

## Update 13-06-2017
- Upgraded to the v2.0 version
- Added BPR-max and TOP1-max losses for cutting edge performance (coupled with additional sampling +30% in recall & MRR over the base results)
- Sacrificed some speed on CPU for faster GPU execution

## Update 22-12-2016
- Fixed cross-entropy unstability. Very small predicted scores were rounded to 0 and thus their logarithm became NaN. Added a small epsilon (1e-24) to all scores before computing the logarithm. I got better results with this stabilized cross-entropy than with the TOP1 loss on networks with 100 hidden units.
- Added the option of using additional negative samples (besides the default, which is the other examples in the minibatch). The number of additional samples is given by the n_sample parameter. The probability of an item choosen as a sample is supp^sample_alpha, i.e. setting sample_alpha to 1 results in popularity based sampling, setting it to 0 results in uniform sampling. Using additional samples can slow down training, but depending on your config, the slowdown might not be noticable on GPU, up to 1000-2000 additional samples.
- Added an option to training to precompute a large batch of negative samples in advance. The number of int values (IDs) to be stored is determined by the sample_store parameter of the train function (default: 10M). This option is for the additional negative samples only, so only takes effect when n_sample > 0. Computing negative samples in each step results in very inefficient GPU utilization as computations are often interrupted by sample generation (which runs on the CPU). Precomputing samples for several steps in advance makes the process more efficient. However one should avoid setting the sample store too big as generating too many samples takes a long time, resulting in the GPU waiting for its completion for a long time. It also increases the memory footprint.

## Update 21-09-2016
- Optimized code for GPU execution. Training is ~2x faster now.
- Added retrain functionality.
