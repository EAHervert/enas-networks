# ENAS DHDN

This library contains the code to run the DHDN model and all the ENAS variations of said model. 

We have that the model has the following hyperparameters:
* DRC Block
  * Number of Convolutions 
* Kernel Size 
  * 3x3
  * 5x5
* Pre-Processor
  * Converts image of 3 channels to stack of processed frames
* Encoder Component
  * DRC Pairs
  * Downsample Method
    * Max Pooling (Default)
    * Average Pooling
    * 2x2 Convolution
* Bottleneck
  * Special Case of the DRC block
* Decoder
  * DRC Pairs
  * Upsampling Method
    * Pixel Shuffling (Default)
    * Bilinear Interpolation
    * Transpose Convolution

The code is of the format:
* DRC Block
* Shared DHDN (Graph Model variant of the DHDN)
