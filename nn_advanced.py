def lrfn(rampup_epochs = 5
        ,max_lr = .001,
        start_lr = 0.00001,
        min_lr = 0.00001,
        sustain_epochs = 0,
        exp_decay = 0.8):
  '''
  Closure to ramp up and then ramp down the training of our model.
  You first provide all the relevant params and then you will get the inner
  closure which you can put into the learning rate scheduler

  Parameters
  ---------------------------
  start_lr : The starting learning rate
  min_lr: The minimum learning rate that the model should go to
  max_lr: The maximum learning rate that the model should achieve
  sustain_epochs = epochs for which the max_lr should be sustained
  rampup_epochs = Epochs till which the learning rate should go up
  exp_decay = How fast should the lr decay after it has achieved its max
  '''
  def inner(epoch):
    if epoch < rampup_epochs:
      return (max_lr - start_lr)/rampup_epochs * epoch + start_lr
    elif epoch < rampup_epochs + sustain_epochs:
      return max_lr
    else:
      return (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr
  
  return inner

import tensorflow as tf
import os
def tpu_strat():
  """
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
  tf.config.experimental_connect_to_cluster(tpu)
  tf.tpu.experimental.initialize_tpu_system(tpu)
  tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
  """
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
  tf.config.experimental_connect_to_cluster(tpu)
  tf.tpu.experimental.initialize_tpu_system(tpu)
  tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
  return tpu_strategy