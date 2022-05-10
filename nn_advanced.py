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

# Regulatisation of weights
# from tensorflow.keras import regularizers
# model = keras.Sequential([
# layers.Dense(16,
# kernel_regularizer=regularizers.l2(0.002),
# activation="relu"),
# layers.Dense(16,
# kernel_regularizer=regularizers.l2(0.002),
# activation="relu"),
# layers.Dense(1, activation="sigmoid")
# ])
# model.compile(optimizer="rmsprop",
# loss="binary_crossentropy",
# metrics=["accuracy"])
# * Weights will also become a part of the loss function and accordingly with l2 and l1 losses
# But these kind of regularisation is only helpful for smaller models, larger models tend to be so big and bulky these thigns don’t matter to them and for them we use dropout

# Custom Callback
import matplotlib.pyplot as plt
from tensorflow import keras
class LossHistory(keras.callbacks.Callback):
  def on_train_begin(self, logs):
    self.per_batch_losses = []
  def on_batch_end(self, batch, logs):
    self.per_batch_losses.append(logs.get("loss"))
  def on_epoch_end(self, epoch, logs):
    plt.clf()
    plt.plot(range(len(self.per_batch_losses)), self.per_batch_losses,
    label="Training loss for each batch")
    plt.xlabel(f"Batch (epoch {epoch})")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"plot_at_epoch_{epoch}")
    self.per_batch_losses = []
# on_epoch_begin(epoch, logs)
# on_epoch_end(epoch, logs)
# on_batch_begin(batch, logs)
# on_batch_end(batch, logs)
# on_train_begin(logs)
# on_train_end(logs)


# Custom training loops
# model = get_mnist_model()
# loss_fn = keras.losses.SparseCategoricalCrossentropy()
# optimizer = keras.optimizers.RMSprop()
# metrics = [keras.metrics.SparseCategoricalAccuracy()]
# loss_tracking_metric = keras.metrics.Mean()
# def train_step(inputs, targets):
# with tf.GradientTape() as tape:
# predictions = model(inputs, training=True)
# loss = loss_fn(targets, predictions)
# gradients = tape.gradient(loss, model.trainable_weights)
# optimizer.apply_gradients(zip(gradients, model.trainable_weights))
# logs = {}
# for metric in metrics:
# metric.update_state(targets, predictions)
# logs[metric.name] = metric.result()
# loss_tracking_metric.update_state(loss)
# logs["loss"] = loss_tracking_metric.result()
# return logs