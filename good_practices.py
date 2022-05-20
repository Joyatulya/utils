#? This is for the initial cell and is used to se the colors accordingly
# and can be used later as needed
# mpl.rcParams['figure.figsize'] = (12, 10)
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# * In case of classification and binaries, this is able to see the 
#  different labels and see what percentage we have of each
# neg, pos = np.bincount(raw_df['Class'])
# total = neg + pos
# print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
#     total, pos, 100 * pos / total))

from tensorflow import keras

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

"""
One very Interesting thing that I read was about bias/weight initialisation, this approach says that often we before hand
have some idea that the predictions are going to turn out as we already have some of the class probabilities.

So when we start training we beforehand assign some weights to the the outputs, so in the initial epochs , our model
dosent have to learn these basic relationships, and can right on get onto learning the appropriate things. This also helps mitigate the hockey curve that we see which is usually not needed
"""
import tensorflow as tf
def auto_select_accelerator():
    """
    For selecting tpu, just in case, so that it really helps us.
    This returns the strategy with the TPU or with the GPU
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    
    return strategy

def build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg'):
    def decode(path):
        file_bytes = tf.io.read_file(path) # Reads and outputs the entire contents of the input filename.

        if ext == 'png':
            img = tf.image.decode_png(file_bytes, channels=3) # Decode a PNG-encoded image to a uint8 or uint16 tensor
        elif ext in ['jpg', 'jpeg']:
            img = tf.image.decode_jpeg(file_bytes, channels=3) # Decode a JPEG-encoded image to a uint8 tensor
        else:
            raise ValueError("Image extension not supported")

        img = tf.cast(img, tf.float32) / 255.0 # Casts a tensor to the type float32 and divides by 255.
        img = tf.image.resize(img, target_size) # Resizing to target size
        return img
    
    def decode_with_labels(path, label):
        return decode(path), label
    
    return decode_with_labels if with_labels else decode

def build_augmenter(with_labels=True):
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        return img
    
    def augment_with_labels(img, label):
        return augment(img), label
    
    return augment_with_labels if with_labels else augment

def build_dataset(paths, labels=None, bsize=32, cache=True,
                  decode_fn=None, augment_fn=None,
                  augment=True, repeat=True, shuffle=1024, 
                  cache_dir=""):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)
    
    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)
    
    AUTO = tf.data.experimental.AUTOTUNE
    slices = paths if labels is None else (paths, labels)
    
    dset = tf.data.Dataset.from_tensor_slices(slices)
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    dset = dset.cache(cache_dir) if cache else dset
    dset = dset.map(augment_fn, num_parallel_calls=AUTO) if augment else dset
    dset = dset.repeat() if repeat else dset
    dset = dset.shuffle(shuffle) if shuffle else dset
    dset = dset.batch(bsize).prefetch(AUTO) # overlaps data preprocessing and model execution while training
    return dset