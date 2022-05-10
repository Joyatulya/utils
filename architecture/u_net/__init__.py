from tensorflow.keras.layers import *
from tensorflow.keras import layers

def if_dropout(layer,dropout):
  if dropout:
    return layers.Dropout(dropout)(layer)
  return layer
def unet(input_size=(256,256,1), dropout: int | None = 0.2):
  """
  For building a UNET architecture model which is 5 levels deep and
  using MaxPooling and Dropout

  Arguments:
  ---

    input_shape : Input shape for the model
    dropout (None | Int) : If dropout should be used, if yes then how much
  """
  inputs = Input(input_size)
  
  # Layer 1 and Level 1
  # As you can see we start here with 32 filters
  conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
  conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  pool1 = if_dropout(pool1,dropout)

  """ 
  And on consecutive layers/levels, through MaxPooling, the stuff is downsampled
  And one important thing to realise here is that, this makes a lot of sense here
  is that we used MaxPooling, so that the brightest feature of all comes out.

  There can be more expoeriments with Average Poolin/Min Pooling to see if the darkest
  pixel might be illuminated which can be of some sense. But it might just convert all of
  our images to zero

  We are doubling the filters on deeper layers so that we are able to extract finer and finer details, also
  we are naming each of the layers differently so as to join them further down
    
    """
  # Layer 2 and Level 2
  conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
  conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  pool2 = if_dropout(pool2,dropout)

  # Layer 3 and Level 3
  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  pool3 = if_dropout(pool3,dropout)

  # Layer 4 and Level 4
  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
  pool4 = if_dropout(pool4,dropout)

  # Layer 5 and Level 5
  conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
  conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
  conv5 = if_dropout(conv5,dropout)

  """
  So, our model is 5 levels deep and we have reached that level, now we are going to start going up
  by concatenating the respective levels and then tconv and conv them without the maxpooling but instead
  with strides of 2 in the tconv layers as in them it leads to upsampling

  Also, in the last layer, we did not to MaxPooling as we initiated the upward sequence from here.
  And we have concatenated not with the deepest layer, the deepeset layer remains as such but instead we concatenated
  with the layer above, 4 in this case, which can be seen by conv4 in the concatenation layer.

  We concatenate in the 3rd axis, which is the last one I think, and this forms a new image with
  both convoluted and shortcut data.
  """
  # Layer 6 and Level 4
  up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
  conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
  conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
  
  # Layer 7 and Level 3
  up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
  conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
  conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

  # Layer 8 and Level 2
  up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
  conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
  conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

  # Layer 9 and Level 1
  up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
  conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
  conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

  # Output
  conv10 = Conv2D(1, (1, 1), dtype = 'float32',activation='sigmoid')(conv9)

  return Model(inputs=[inputs], outputs=[conv10])