# GRU4Rec in Keras

This repository offers an implementation of the "Session-based Recommendations With Recurrent Neural Networks" paper (https://arxiv.org/abs/1511.06939) using the Keras framework, tested with TensorFlow backend.

A script that interprets the MovieLens 20M dataset as if each user's history were one anonymous session (spanning anywhere from months to years) is included. Our implementation presents comparable results to those obtained by the original Theano implementation offered by the GRU4Rec authors, both over this new domain and also one of the original datasets used in the paper: 2015 RecSys Challenge dataset (RSC15).

Aditionally, a script can be found for determining a dataset's Dwell Time information, as seen on "Incorporating Dwell Time in Session-Based Recommendations with Recurrent Neural Networks" (http://ceur-ws.org/Vol-1922/paper11.pdf). Used with the RSC15 dataset, augmentation results can be reproduced, although we have not been able to replicate the final reported performance metrics.

Credit goes to yhs-968 for his parallel-batch data loader, as shown in his pyGRU4REC repository (https://github.com/yhs-968/pyGRU4REC).

To train the RNN model from scratch: 

```python model/gru4rec.py --train-path path/to/train.csv --dev-path path/to/validation.csv --test-path path/to/test.csv```.

To resume training from a checkpoint, add ```--resume path/to/model_weights.h5``` to the previous command.

To run the Dwell Time augmentation process: ```python preprocess/extractDwellTime.py```.

Future work contemplates incorporating dwell time in an online manner to the model, hoping to leverage said information in the learning process, instead of in a previous preprocessing stage.

## Requirements

The code has been tested with Python 3.6.8, using the following versions of the required dependencies:
- numpy == 1.16.3
- pandas == 0.24.2
- tqdm == 4.32.1
- tensorflow == 1.11.0rc1
- keras == 2.2.4
