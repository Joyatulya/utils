"""
A ResNet model builder which is adapted for use in a FCN for 
the purposes of semantic segmentation.

This model will forrm the backbone of the model, ie - this 
will work as the feature extractor which will tehn be fed into the 
feature pyramid

"""

import numpy as np
from tensorflow import keras
from keras import layers

from model import conv_layer
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------

def resnet_layer(
  inputs,
  num_filters = 16,
  kernel_size = 3,
  strides = 1,
  activation = 'relu',
  batch_normalisation = True,
  conv_first = True,
):

  """
  2D Conv - BN - Activation stage Builder
  """

  conv = layers.Conv2D(
    num_filters,
    kernel_size = kernel_size,
    strides = strides,
    padding = 'same',
    kernel_initializer= 'he_normal',
    kernel_regularizer= keras.regularizers.l2(1e-4)
  )

  def bn_relu(input):
    x = input
    if batch_normalisation:
      x = layers.BatchNormalization()(x)
    if activation is not None:
      x = layers.Activation(activation)(x)
    return x

  x = inputs
  if conv_first:
    x = conv(x)
    x = bn_relu(x)
  else:
    x = bn_relu(x)
    x = conv(x)
  
  return x

def resnet_v1(input_shape,
             depth : int = 20,
             num_classes:int = 2) -> keras.models.Model:
  """
  ResNet Version 1 builder

  Here I'll be making my first ResNet, its going to be some task.

  All I know about resnet that it has a few blocks which I made one above and all these blocks
  have residual connections from one layer to another, just like skip connectons

  So this has stages of 2 X (3 X 3) stage layers above in which the last ReLu is after the shortcut connection.

  A ResNet has multiple stages and at the beginning of each stage, the feature map size is haved with ofcourse
  a double strided conv. Also the number of filters are doubled

  Within each stage, the layers have same number of filters
              
              Size  Filters
  - Stage 0 : 32x32, 16
  - Stage 0 : 16x16, 32
  - Stage 0 : 08x08, 64

  Arguments:
  ======================
    input_shape : Input shape of the image
    depth (int) : Number of core conv layers. *Depth should be 6n+2 (eg 20, 32, 44 in [a])*
    num_classes (int) : Number of classes for which you are aimimg for

  Returns:
  ====================
    model (keras.models.Model) : Keras model instance
  """

  if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')

  # Starting the model definition here
  num_filters = 16
  num_res_blocks = int((depth - 2) / 6)

  inputs = layers.Input(shape= input_shape, name = "Input")
  # Irrespective of the size we start the first layer with 16 filters
  x = resnet_layer(inputs)

  # Instantiating the stage of residual units

  for stage in range(3):
    for res_block in range(num_res_blocks):

      strides = 1 # Here we reset the stride irrespective of our locatio in the model
      
      if stage > 0 and res_block == 0: # Looking for first layer but not in the first stage
        strides = 2 # This means we downsample after the first stage

      y = resnet_layer(inputs = x, 
                       num_filters= num_filters,
                       strides = strides
                        )
      y = resnet_layer(inputs = y, 
                       num_filters= num_filters,
                       activation=None
                        )
      
      if stage > 0 and res_block == 0:
        x = resnet_layer(inputs= x,
                         num_filters=num_filters,
                         strides=strides,
                         activation=None,
                         batch_normalisation=None)

      x = layers.Add()([x,y])
      x = layers.Activation('relu')(x)
    
    num_filters *= 2
  
  # Feature Maps
  outputs = feature_pyramid(x,3)

  # Instantiate model
  name = 'ResNet%dv1' % (depth)

  return keras.models.Model(inputs = inputs,
                            outputs = outputs,
                            name = name)

  return model


def feature_pyramid(x,
              n_layers):
  """
  Generates a feature pyramid from the output of the last layer of the backbone network. Then backbone network can be
  any of the ResNets but I havent tried the other ones.

  This will contain multiple conv2d layers along with BN and relu activation followed by another Conv2D layers.

  Arguments:
    x (tensor): Output feature maps extracted from the feature extractor which is our backbone layer.
    n_layers (int): Number of additional pyramid layers that we need inside

  Return:
    outputs (list): Outputs a feature pyramid with different details of the data from small conv layers and large conv layers
    and this needs to be fed into the upsampling and the decoder portion of our NN
  """

  # This portion is for first averaging the outputs before feeding into the pyramid. We arent calculating anything as of now
  # we are just building up our model to be trained afterwards
  outputs = [x]
  conv = layers.AveragePooling2D(pool_size = 2, name = 'pool1')(x)
  outputs.append(conv)

  prev_conv = conv
  n_filters = 512
  for i in range(n_layers - 1):
    postfix = '_layer' + str(i + 2)
    conv = conv_layer(prev_conv,
                      filters = n_filters,
                      kernel_size = 3,
                      strides = 2,
                      use_maxpool = False,
                      postfix = postfix)
    outputs.append(conv)
    prev_conv = conv
  
  return outputs