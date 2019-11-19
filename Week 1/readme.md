Assignment 1:

Definitions:

Convolution:
In the context of image processing, Convolution is the process of extracting relevant features from an image/video that can be used downstream for tasks like image labelling, object recognition.

Filters/Kernels:
Filters are the matrices that convolve over images to produce maps of relevant features. The filters may be trained to perform image processing tasks.

Epochs:
A model is said to have been trained for one 'Epoch', if it has seen every example in the training set once.

1x1 Convolution:
The process of convolving over an image using a 1x1 filter (mainly used for compressing number of channels/ reducing redundancies) to produce a resultant feature of the same dimension (but fewer channels).

3x3 Convolution:
The process of convolving over an image using a 3x3 filter, to produce a resultant feature set required for accomplishing image/video processing tasks like object recognition, image labelling, etc.

Feature Map:
The result of convolutions, a feature map is a set of channels with the same kind of information (like vertical/horizontal edges) in each channel.

Activation Function:
The final function used to transform the result of addition/multiplication operations in convolutions, generally used to introduce non-linearity to the model.

Receptive Field:
The number of pixels of the previous layer (local receptive field), or the original image (global receptive field) whose information is contained in one pixel of the reference layer.
