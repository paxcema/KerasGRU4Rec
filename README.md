# GRU4Rec in Keras

## 🚩 Important 🚩

There is now an official implementation of GRU4Rec in both TensorFlow and PyTorch. We recommend that users opt for these updated official implementations, instead:
1. [TensorFlow](https://github.com/hidasib/GRU4Rec_Tensorflow_Official)
2. [PyTorch](https://github.com/hidasib/GRU4Rec_PyTorch_Official)

As noted [in this paper](https://arxiv.org/pdf/2307.14956.pdf), our implementation currently has some important drawbacks: 
1. No support for BPR-max loss
2. No support for item embeddings (either separate or shared)
3. No support for negative sampling, which leads to poor scaling for large datasets

Ultimately, it has been shown this implementation does not achieve the same performance as the original model (see table 6 on the paper above for more details).

We intend to fix these in the near future, but we urge users to beware and instead consider official implementaions linked above. 

## Summary

This repository offers an implementation of the "Session-based Recommendations With Recurrent Neural Networks" paper (https://arxiv.org/abs/1511.06939) using the Keras framework, tested with TensorFlow backend.

A script that interprets the MovieLens 20M dataset as if each user's history were one anonymous session (spanning anywhere from months to years) is included. Our implementation presents comparable results to those obtained by the original Theano implementation offered by the GRU4Rec authors, both over this new domain and also one of the original datasets used in the paper: 2015 RecSys Challenge dataset (RSC15).

Aditionally, a script can be found for determining a dataset's Dwell Time information, as seen on "Incorporating Dwell Time in Session-Based Recommendations with Recurrent Neural Networks" (http://ceur-ws.org/Vol-1922/paper11.pdf). Used with the RSC15 dataset, augmentation results can be reproduced, although we have not been able to replicate the final reported performance metrics.

Credit goes to yhs-968 for his parallel-batch data loader, as shown in his pyGRU4REC repository (https://github.com/yhs-968/pyGRU4REC).

## Instructions

To train the RNN model from scratch: 

```python model/gru4rec.py --train-path path/to/train.csv --dev-path path/to/validation.csv --test-path path/to/test.csv --epoch n_epochs```.

To resume training from a checkpoint, add ```--resume path/to/model_weights.h5``` to the previous command.

To run the Dwell Time augmentation process: ```python preprocess/extractDwellTime.py --train-path path/to/train.csv --output-path path/to/augmented_train.csv```.

Future work contemplates incorporating dwell time in an online manner to the model, hoping to leverage said information in the learning process, instead of in a previous preprocessing stage.

## Changelog

[23/8/2020] Updated backend to TensorFlow 2.3.0

[04/09/2021] Updated backend to TensorFlow 2.6.0

[24/03/2023] Updated backend to TensorFlow 2.11.1

## Requirements

The code has been tested with Python 3.6.8, using the following versions of the required dependencies:
- numpy == 1.18.5
- pandas == 1.0.5
- tqdm == 4.41.1
- tensorflow == 2.6.0
- keras == 2.4.3
- matplotlib == 3.3.1
