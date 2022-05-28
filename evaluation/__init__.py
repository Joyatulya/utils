__all__ = ['METRICS']
from cv2 import threshold
from tensorflow import keras

def statistical_metrics(threshold = 0.5):
  METRICS = [
        keras.metrics.TruePositives(name='tp', thresholds = threshold),
        keras.metrics.FalsePositives(name='fp', thresholds = threshold),
        keras.metrics.TrueNegatives(name='tn', thresholds = threshold),
        keras.metrics.FalseNegatives(name='fn', thresholds = threshold), 
        keras.metrics.BinaryAccuracy(name='accuracy',),
        keras.metrics.Precision(name='precision', thresholds = threshold),
        keras.metrics.Recall(name='recall', thresholds = threshold),
        keras.metrics.AUC(name='auc'),
        keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
  ]

  return METRICS