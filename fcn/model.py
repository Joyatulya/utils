import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

def conv_layer(inputs,
              filters = 32,
              kernel_size = 3,
              strides = 1,
              use_maxpool = True,
              postfix = None,
              activation = None):
  """
  Helper function which is used to build a Conv2D-BN-ReLU layer along with optional max-pool layer
  """

  x = layers.Conv2D(filters = filters,
                    kernel_size = kernel_size,
                    strides = strides,
                    name = 'conv_' + postfix,
                    padding = 'same')(inputs)
  x = layers.BatchNormalization(name = 'bn_' + postfix)(x)
  x = layers.Activation('relu', name = 'relu_' + postfix)(x)
  if use_maxpool:
    x = layers.MaxPooling2D(name = 'pool_' + postfix)(x)
  return x

def tconv_layer(inputs,
                filters = 32,
                kernel_size = 3,
                strides = 2,
                postfix = None
                ):
  """
  Helper function which is used to build a Transpose Conv2D-BN-ReLU layer along with optional max-pool layer
  """
  x = layers.Conv2DTranspose(filters = filters,
                    kernel_size = kernel_size,
                    strides = strides,
                    name = 'transpose_conv_' + postfix,
                    padding = 'same')(inputs)
  x = layers.BatchNormalization(name = 'transpose_bn_' + postfix)(x)
  x = layers.Activation('relu', name = 'transpose_relu_' + postfix)(x)
  return x

def build_fcn(input_shape,
              backbone,
              n_classes = 4):
  """
  Function which will be used to build the FCN model which inlcudes
    - The backbone model
    - The feature pyramid
    - The upsampling
    - Concatenation
    - Upsampling with transposed_Conv2D layers

  Arguments:
  ----------------
    input_shape (Tuple[Int]): The input shape of the image to be expected by our network
    backbone (Model): This will be the backbone network which will work as the initial feature extractor on top of which 
    the rest of the model will be built
    n_classes (int): The different areas into which we want to segment the images fed into the network. This should ideally include
    the background but should be used with your own descretion.
  """

  inputs = layers.Input(shape = input_shape)
  features = backbone(inputs)

  # Here I think we are separating our output to be bypass the feature pyramid and to be
  # concatenated with it afterwards
  main_feature = features[0]
  features = features[1:]
  out_features = [main_feature]
  feature_size = 8
  size = 2

  """The other half of the features pyramid apart from the helper function before.
  Here well do the sampling, and the upsampling to restore the dimensions of our feature
  till the 1/4 of the image size. 
  
  The 1/4 here is I think because of how resnet my output the feature vector and the dimensions to which its reduced to
  """

  for feature in features:
    postfix = 'fcn_' + str(feature_size)
    feature = conv_layer(feature,
                         filters = 256,
                         use_maxpool = False,
                         postfix = postfix)
    postfix += '_up2D'
    feature = layers.UpSampling2D(size = size,
                                  interpolation = 'bilinear',
                                  name = postfix)(feature)
    size *= 2
    feature_size *= 2
    out_features.append(feature)

  # By now we have out_features array which have at the start the main_feature and afterwards have all the upsampled features
  # through the above loop in which they were also given the triple layer of feature pyramid

  x = layers.Concatenate()(out_features)

  # Now we have to upsample it more but this time now using the transpose_conv layers for which we have already written the helper
  # function

  x = tconv_layer(x, filters=256, postfix = 'up_x2')
  x = tconv_layer(x, filters=256, postfix = 'up_x4')

  """
  Now we are at the end of constructing our model. This is not equal to building and fitting the model as that will happend in a separate
  function with the compile method and we also have to see how to give it the data that it wants.

  In the end of our model we will generate our pixel_wise classifier. Remember how its done, depending on the number of classes into
  which we want our model to segment, we provide that dimensional one_hot vector to each pixel and then with the magic of softmax on 
  each pixel, we discern to which of the classes does our model belong to.

  The last one is also built with the help a transpose_Conv2d layer and here the activation is in a different layer, so dont
  be thrown off by that. Apart from that, in case of binary classification like that for lung or not lung, I think we can
  also use a sigmoid function in the end. This can be experimented afterwards
  
  I am unable to understand though that how filters inside a tconv layer will be able to be used as classifier model. I'll
  have to read into the documentation of this.
  """

  x = layers.Conv2DTranspose(filters = n_classes,
                             kernel_size = 1,
                             strides = 1,
                             padding = 'same',
                             kernel_initializer='he_normal',
                             name = 'pre_activation')(x)
  outputs = layers.Softmax(name = 'segmentation')(x)

  model = keras.models.Model(inputs,outputs,name = 'fcn')

  return model