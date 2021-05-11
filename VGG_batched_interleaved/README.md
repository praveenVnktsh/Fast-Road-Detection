## VGG - Batched Interleaved Model

Using N images to train the ConvLSTM, we train this model over the lane segementation dataset.
Here, instead of the Resnet FE, we USE VGG 11 and VGG 16 models.

This is our model:

<p align="center">
  <img width="600" src="../extras/trainingpipeline.png">
</p>

This code is written in lightening and PyTorch.
