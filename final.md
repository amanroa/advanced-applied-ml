---
layout: page
title: "Final"
permalink: /final/
---

# Using ResNet-50 for Bird Song Identification
## By Aashni Manroa

# Introduction

Identifying birds based on bird song has been heavily researched. It is
not as simple as one might think - it has been shown that bird song
varies, even with individual birds. For example, male birds show more
creativity with their songs when they are singing alone, as opposed to
singing to a female bird . This variability in song makes it difficult
to create a highly accurate ML model. In addition to variability in the
songs, background noise and overlapping bird song all make it difficult
to identify birds.

Researchers have tried multiple models to overcome these obstacles: deep
convolutional neural networks , support vector machines, and hidden
Markov models have all been attempted in the quest to create a perfect
bird identification model.

Based on my research, a deep convolutional neural network (DCNN) with
residual blocks is the most accurate method available for identifying
birds . I decided to use this information to create my own model. The
goal was to create a model that achieved an accuracy greater than 50% on
the data.

# Data Description

When searching for datasets for this topic, I came across two viable
options on Kaggle. The first is titled *BirdCLEF 2024*, which contains
100 audio clips for about 200 species of birds . This comprehensive
dataset, however, was a bit too large - about 23 GB. I felt that my
computer would not be able to process all of this data, so I searched
for a smaller dataset.

The dataset that I chose is called *Bird Song Dataset*, and it was
uploaded to Kaggle by a user named vinayshanbhag . This dataset contains
5,422 audio files for five different species of birds: the American
Robin, Bewick’s Wren, the Northern Cardinal, the Northern Mockingbird,
and the Song Sparrow. This means that there are about 1,100 audio files
per species. Each audio file is 3 seconds long. This dataset also
includes metadata for each recording, such as the longitude, latitude,
and altitude in which it was taken. I decided not to include these
extraneous variables in my model, as my only focus was to train the
model on audio. Finally, the author of the dataset stated that they only
included examples of bird songs, rather than all of the types of noise a
bird can make - alarm calls, trilling, contact calls, etc. This is to
maintain the simplicity of the dataset.

<figure id="fig:enter-label">
<img src="color_spec.png" />
<figcaption>A spectrogram of the words "nineteenth century" being spoken
<span class="citation" data-cites="spectrogram"></span>.</figcaption>
</figure>

# Methods

## Pre-Processing Methods

There were multiple possible methods for using audio to train my model,
such as a time series or a CNN. With a time series model, I would use
the audio clips in their current format to train the model. But with a
CNN model, I could convert the audio clips into spectrograms.
Spectrograms, as shown in Figure 1, are a 2-D representation of audio
frequencies as they vary with time . After converting the audio to
images, I could train a CNN on the converted images. I thought that the
latter method was interesting, so I chose to use a CNN model.

To pre-process my audio files, I first created two directories, train
and test. I then created sub-directories for each species within the
train and test folders. Using *sklearn’s train-test-split* library, I
split the x data (audio file names) and y data (bird name), with the
test set being 30% of the full dataset.

Next, I iterated through the *xtrain* and *xtest* sets used the Librosa
library to create spectrograms out of the audio. To make my algorithm
less complex and reduce the computational requirements needed, I chose
to make my spectrograms grayscale. I then saved the image - which was
515 x 389 pixels - to it’s respective folder. The pre-processing
component of my model took much longer than I predicted. Figure 2 shows
an example of a spectrogram that I used for each species.

<figure id="fig:enter-label">
<img src="gray_spec.png" />
<figcaption>Examples of spectrograms used for each species.</figcaption>
</figure>

Additionally, I used ImageDataGenerators to modify the images in the
train class by rotating, zooming, stretching, and flipping them. By
doing this, it made my model more robust.

## Machine Learning Methods

After pre-processing, I had to decide which CNN to use for my model. I
did some research on the different types of CNNs, and came across a few
options.

My first thought with the model type was that the grayscale images
reminded me of the MNIST data. So, I looked into the LeNet-5
architecture, which is a model that is commonly used for the MNIST data.
However, the LeNet-5 model takes in images of a low resolution . Due to
the large resolution of my spectrograms, I couldn’t resize them to be
that small, or they would lose all of their identifying data. Figure 3
is an example of a spectrogram after being resized to 32 x 32 pixels. As
shown in the picture, it is much too small for my model to distinguish
any patterns.

<figure id="fig:enter-label">
<img src="bad_res.png" />
<figcaption>A low resolution spectrogram that I would have used for the
LeNet-5 model.</figcaption>
</figure>

After ruling out LeNet-5, I looked at models we covered in class. I
specifically looked at AlexNet. This model was able to train on higher
quality images than LeNet-5, but it only has 8 layers. I wanted to find
a model that had enough layers to recognize fine patterns, but one that
isn’t so complex that my computer wouldn’t be able to run it.

That is when I found ResNet-50. ResNet-50 is a Deep CNN, meaning it has
multiple layers. It has 50 layers, and can provide a higher accuracy
than AlexNet. It is able to handle complex datasets, and is able to
learn intricate patterns. The way that ResNet50 is able to handle
complex datasets is with it’s unique feature of residual blocks.
Residual blocks, as shown in Figure 4, occur after every 2 convolutions,
and we bypass the layer in between. This is used to reduce the vanishing
gradient problem. The larger architecture of ResNet-50 ends up looking
something like Figure 5, which is a diagram of ResNet-34 . We can see
the multiple residual blocks that are implemented throughout the model.
Another benefit of ResNet-50 is that it takes higher quality images as
input - around 224 x 224. I felt that there wasn’t as much data loss
with this image size, so I decided to use this model in my project.

<figure id="fig:enter-label">
<img src="residual.png" />
<figcaption>A diagram of one residual block in the ResNet-50
architecture.</figcaption>
</figure>

<figure id="fig:enter-label">
<img src="resnet.png" />
<figcaption>Right: ResNet-34 architecture, Middle: Plain network with 34
layers, Left: Architecture for a VGG-19 model, which has 19
layers.</figcaption>
</figure>

## Application of Methods and Validation Procedure

To use the ResNet50 model, I imported Tensorflow’s version of it. Once I
did that, I created a base model using the ResNet50 class. An important
addition - I added an arugment in the base model’s initialization titled
*"weights = ’imagenet’"*. This initialized the ResNet model with some
base weights from the ImageNet database, which I thought would be
helpful. This way, the model could spend more time on the fine details
rather than trying to understand the basic shapes of the spectrograms .

Next, I added a GlobalAveragePooling2D layer. This simplified the output
by creating a single average number per feature discovered. Finally, I
added a Dense layer with softmax activation, just to make sure the
output would represent the 5 y classes (bird species). A high level
diagram of my model is shown in Figure 6.

After creating my model, I trained it for 10 epochs. After training for
10 epochs, I got an accuracy of 57.59% on the validation set, and 76.65%
on the training set. Unfortunately, that might indicate overfitting.

<figure id="fig:enter-label">
<img src="model.png" />
<figcaption>A high level image of my model</figcaption>
</figure>

# Discussion and Inferences

To see where my model went wrong, I decided to create a confusion
matrix, which mapped out the predicted labels as compared to the actual
labels. This is shown in Figure 7. As we can see, the model most often
mislabeled the other 4 species as a Song Sparrow - or if it was looking
at an image from a Song Sparrow, it thought that it was an image from a
Bewick’s Wren.

This is interesting, and it could be due to a number of reasons. First,
the spectrogram quality. These spectrograms could contain a lot of
background or ambient noise, and that could confuse the model. Second,
it could be because I didn’t have enough data to begin with. With each
species only having 1100 samples, that means that the model has to train
on only 770 images. I believe this might be the root issue. Another
issue I could think of is that I chose a far too complex model for this
smaller dataset.

<figure id="fig:enter-label">
<img src="confusion.png" />
<figcaption>Confusion matrix for my model.</figcaption>
</figure>

# Conclusion

In conclusion, this project was very interesting, and I was eager to
explore the world of bird identification. However, my model is far from
perfect. In the future, I might try using a Time Series model, rather
than a CNN. Or, I could try using my ResNet-50 model with the much
larger BirdCLEF data, to see if it reduces the overfitting.

# Acknowledgment

I would like to thank Professor Vasiliu for aiding me with my model and
giving me ideas for improvements to this project.

1

Kao, M. and Brainard, M. (2006). Lesions of an avian basal ganglia
circuit prevent context-dependent changes to song variability. Journal
of Neurophysiology, 96(3), 1441-1455.
https://doi.org/10.1152/jn.01138.2005

Zhang, F., Chen, H., & Xie, J. (2021). Bird species identification using
spectrogram based on multi-channel fusion of dcnns. Entropy, 23(11),
1507. https://doi.org/10.3390/e23111507

Arriaga, J. G., Kossan, G., Cody, M. L., Vallejo, E. E., & Taylor, C.
(2013). Acoustic sensor arrays for understanding bird communication.
identifying cassin’s vireos using svms and hmms. Advances in Artificial
Life, ECAL 2013. https://doi.org/10.7551/978-0-262-31709-2-ch120

BirdCLEF 2024 \| Kaggle. (n.d.).
https://www.kaggle.com/competitions/birdclef-2024

bird song data set. (2020, June 22). Kaggle.
https://www.kaggle.com/datasets/vinayshanbhag/bird-song-data-set

Huilgol, P. (2023, August 9). Top 4 Pre-Trained Models for Image
Classification with Python Code. Analytics Vidhya.
https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/

Wikipedia contributors. (2024, May 10). Spectrogram. Wikipedia.
https://en.wikipedia.org/wiki/Spectrogram

Bangar, S. (2022, June 22). LeNEt 5 Architecture Explained - Siddhesh
Bangar - Medium. Medium.
https://medium.com/@siddheshb008/lenet-5-architecture-explained-3b559cb2d52b
