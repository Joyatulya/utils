"""
Usage Tips
--------------

In my experience testing and debugging these losses, I have some observations that may be useful to beginners experimenting with different loss functions. These are not rules that are set in stone; they are simply my findings and your results may vary.

Tversky and Focal-Tversky loss benefit from very low learning rates, of the order 5e-5 to 1e-4. They would not see much improvement in my kernels until around 7-10 epochs, upon which performance would improve significantly.

In general, if a loss function does not appear to be working well (or at all), experiment with modifying the learning rate before moving on to other options.

You can easily create your own loss functions by combining any of the above with Binary Cross-Entropy or any combination of other losses. Bear in mind that loss is calculated for every batch, so more complex losses will increase runtime
"""

from re import I
import tenorflow as tf
from tensorflow import keras
import keras.backend as K
def dice_coeff(y_true, y_pred, smooth = 1e-4):
  """
  Dice Coefficient, this tells about the level of overlap between
  the predicted image and the mask.

  This is better than other losses in case of segmentation, where usually
  the backgroung >> foreground.

  So instead of decereasing the loss by unfair means, but it focuses
  on the foreground
  """
  # Flatten
  y_true_f = tf.reshape(y_true, [-1])
  y_pred_f = tf.reshape(y_pred, [-1])
  intersection = tf.reduce_sum(y_true_f * y_pred_f)
  score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
  return score

def dice_loss(y_true, y_pred):
  """
  Loss for using the dice coeff as dice_coeff is between 0 - 1, where
  larger is better

  Here, we just 1 - dice_coeff, so as to approach 0 as all loss
  functions usually do
  """
  loss = 1 - dice_coeff(y_true, y_pred)
  return loss

def bce_dice_loss(y_true, y_pred):
  """
  Best of both worlds, where we take into account, both the Binary Crossentropy
  loss and also the dice loss
  """

  loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
  return loss

def IoULoss(targets, inputs, smooth=1e-6):
    
  #flatten label and prediction tensors
  inputs = K.flatten(inputs)
  targets = K.flatten(targets)
  
  intersection = K.sum(K.dot(targets, inputs))
  total = K.sum(targets) + K.sum(inputs)
  union = total - intersection
  
  IoU = (intersection + smooth) / (union + smooth)
  return 1 - IoU

def FocalLoss(targets, inputs, alpha=0.8, gamma=2):    
    """
    Focal Loss was introduced by Lin et al of Facebook AI Research in 2017 as a means of combatting extremely imbalanced datasets where positive cases were relatively rare. Their paper "Focal Loss for Dense Object Detection" is retrievable here: https://arxiv.org/abs/1708.02002. In practice, the researchers used an alpha-modified version of the function so I have included it in this implementation.
    """
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss

def TverskyLoss(targets, inputs, alpha=0.5, beta=0.5, smooth=1e-6):
  """
  This loss was introduced in "Tversky loss function for image segmentationusing 3D fully convolutional deep networks", retrievable here: https://arxiv.org/abs/1706.05721. It was designed to optimise segmentation on imbalanced medical datasets by utilising constants that can adjust how harshly different types of error are penalised in the loss function. From the paper:

  in the case of α=β=0.5 the Tversky index simplifies to be the same as the Dice coefficient, which is also equal to the F1 score. With α=β=1, Equation 2 produces Tanimoto coefficient, and setting α+β=1 produces the set of Fβ scores. Larger βs weigh recall higher than precision (by placing more emphasis on false negatives).

  To summarise, this loss function is weighted by the constants 'alpha' and 'beta' that penalise false positives and false negatives respectively to a higher degree in the loss function as their value is increased. The beta constant in particular has applications in situations where models can obtain misleadingly positive performance via highly conservative prediction. You may want to experiment with different values to find the optimum. With alpha==beta==0.5, this loss becomes equivalent to Dice Loss.
  """
        
  #flatten label and prediction tensors
  inputs = K.flatten(inputs)
  targets = K.flatten(targets)
  
  #True Positives, False Positives & False Negatives
  TP = K.sum((inputs * targets))
  FP = K.sum(((1-targets) * inputs))
  FN = K.sum((targets * (1-inputs)))
  
  Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
  
  return 1 - Tversky

def FocalTverskyLoss(targets, inputs, alpha=0.5, beta=0.5, gamma=1, smooth=1e-6):
  """
  A variant on the Tversky loss that also includes the gamma modifier from Focal Loss.
  """
  #flatten label and prediction tensors
  inputs = K.flatten(inputs)
  targets = K.flatten(targets)
  
  #True Positives, False Positives & False Negatives
  TP = K.sum((inputs * targets))
  FP = K.sum(((1-targets) * inputs))
  FN = K.sum((targets * (1-inputs)))
          
  Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
  FocalTversky = K.pow((1 - Tversky), gamma)
  
  return FocalTversky

def Combo_loss(targets, inputs, eps=1e-9, ALPHA = 0.5, CE_RATIO = 0.5, smooth = 1e-6):
  """
  This loss was introduced by Taghanaki et al in their paper "Combo loss: Handling input and output imbalance in multi-organ segmentation", retrievable here: https://arxiv.org/abs/1805.02798. Combo loss is a combination of Dice Loss and a modified Cross-Entropy function that, like Tversky loss, has additional constants which penalise either false positives or false negatives more respectively.
  """
  targets = K.flatten(targets)
  inputs = K.flatten(inputs)
  
  intersection = K.sum(targets * inputs)
  dice = (2. * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
  inputs = K.clip(inputs, eps, 1.0 - eps)
  out = - (ALPHA * ((targets * K.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * K.log(1.0 - inputs))))
  weighted_ce = K.mean(out, axis=-1)
  combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
  
  return combo