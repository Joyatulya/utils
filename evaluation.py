# K-Fold cross validation
# k = 3
# num_validation_samples = len(data) // k
# np.random.shuffle(data)
# validation_scores = []
# for fold in range(k):
# validation_data = data[num_validation_samples * fold:
# num_validation_samples * (fold + 1)]
# training_data = np.concatenate(
# data[:num_validation_samples * fold],
# data[num_validation_samples * (fold + 1):])
# model = get_model()
# model.fit(training_data, ...)
# validation_score = model.evaluate(validation_data, ...)
# validation_scores.append(validation_score)
# validation_score = np.average(validation_scores)
# model = get_model()
# model.fit(data, ...)
# test_score = model.evaluate(test_data, ...)

# All of the following is for assessing what exactly a conv net sees and what are the different features that are extraccted by the different layers
# inside of a nn

best_model = keras.models.load_model('./oxford_segmentation.keras')
layer_outputs = []
layer_names = []
for layer in best_model.layers:
  if isinstance(layer,(layers.Conv2D,layers.MaxPooling2D)):
    layer_outputs.append(layer.output)
    layer_names.append(layer.name)
layer_outputs
"""
So what exactly is happening here is that we are inputting all theses values as a list to the output of the model, and in the end
what we weill get is the list of activation of all the model activations and will be able to see them in action.

The thing that was putting me off was that this had multiple outputs at the same time and we havent seen that before, but is actually fine an
d i just ahve ot get used to it.
"""
activation_model = keras.Model(inputs = best_model.inputs,outputs = layer_outputs)
img = tf.io.read_file('./Lung Segmentation/test/CHNCXR_0181_0.png')
img = tf.image.decode_image(img,channels = 1)
img = tf.image.resize(img,(256,256))
activations = activation_model.predict(tf.expand_dims(img,0))
plt.imshow(img[:,:,0],cmap = 'viridis')
LAYER = 2
KERNEL = 19
first_layer_activation = activations[LAYER]
the_layer_i_want = tf.cast(first_layer_activation[0,:,:,KERNEL],'float32')
plt.figure(figsize = (10,10))
plt.imshow(the_layer_i_want,cmap='viridis')
first_layer_activation.shape
