# ENAS DHDN

This library contains the code to run the DHDN model and all the ENAS variations of said model. 

We have that the model has the following hyperparameters:
* DRC Block
  * Number of Convolutions 
* Kernel Size
  * Choose from mix of 3x3 and 5x5 
* Pre-Processor
  * Converts image of 3 channels to stack of processed frames
* Encoder Component
  * DRC Pairs and Downsample Method
    * Max Pooling (Default), Average Pooling, 2x2 Convolution
* Bottleneck
* Decoder
  * DRC Pairs and Upsampling Method including
    * Pixel Shuffling (Default), Bilinear Interpolation, Transpose Convolution

The code is of the format:
* Controller - Returns arrays specifying architectures
* DRC - Code that generates the DRC blocks for DHDN
* Resampling - Containing all the resampling blocks (MAX Pooling, ect.)
* Shared DHDN - Graph Model variant of the DHDN
* Training Functions - Functions used in training the shared network given some controller
  * train_loop - Trains the shared network based on controller (if passed)
  * evaluate_model - Finds best architectures based on given models
* Training Networks
  * Train_Shared - Trains the shared network using the train_loop
  * Train_Controller - If controller is not None, will train controller based on shared network
  * Train_ENAS - Trains the shared network and controller to try and find optimal networks
