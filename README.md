# Style-Transfer-using-Deep-Learning
The objective is to apply the style of an image, which we will term as "style image" to a target image while preserving the content of the target image. 

In style transfer, we take two images, one is the content image and the other one is the style image. We blend the two images together so that the output image looks like the content image but painted in style of the style reference image and this is done using tensorflow.

Convolutional neural networks are the most powerful breed of neural networks for image classification and analysis. Neural Style Transfer compose images based on layer activations within CNNs and their extracted features. A CNN is often a collection of several convolutional layers and pooling layers. Convolutional layers are responsible for extracting highly complex features from a given image while the pooling layers discard detailed spatial information that is not relevant for an image classification problem. The effect of this is it helps the CNN to learn the content of a given image rather than anything specific such as color, texture, and so on. As we go deeper into a CNN, the complexity of the features increase and the deeper convolutional layers are often referred to as content representations.

I used the intermediate layers of the model to get the content and style representations of the image. Starting from the network's input layer, the first few layer activations represent low-level features like edges and textures. As you step through the network, the final few layers represent higher-level features—object parts like wheels or eyes. In this case, you are using the VGG19 network architecture, a pretrained image classification network. These intermediate layers are necessary to define the representation of content and style from the images.
These intermediate outputs within our pretrained image classification network allow us to define style and content representations because at a high level, in order for a network to perform image classification (which this network has been trained to do), it must understand the image. This requires taking the raw image as input pixels and building an internal representation that converts the raw image pixels into a complex understanding of the features present within the image.
I also calculated the total loss taking in account the stle loss, content loss and variation loss and the optimization was done on each iteration to reduce the loss.

 The following are the steps involved for Neural Style Transfer:
 1.Visualize the input
 2.Fast Style Transfer using TF-Hub
 3.Define content and style representations
 4.Build the model
 5.Calculate style
 6.Extract style and content
 7.Run gradient descent
 8.Total variation loss
 9.Re-run the optimization
 
 References:
 1.https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398
 2.https://www.tensorflow.org/tutorials/generative/style_transfer
 3.https://www.datacamp.com/community/tutorials/implementing-neural-style-transfer-using-tensorflow
 4.https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
