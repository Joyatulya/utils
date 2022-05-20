"""
Evaluation metrics for multi-label datasets
The evaluation metrics for multi-label classification can be broadly classified into two categories —

 - Example-Based Evaluation Metrics
 - Label Based Evaluation Metrics.

https://medium.datadriveninvestor.com/a-survey-of-evaluation-metrics-for-multilabel-classification-bb16e8cd41cd

"""

# Example Base Evaluation Metrics
# ------------------- & ----------------------

import numpy as np


def emr(y_true, y_pred):
  """
  Tells about how many of the predictions have an exact match
  """
  n = len(y_true)
  row_indicators = np.all(y_true == y_pred, axis = 1) # axis = 1 will check for equality along rows.
  exact_match_count = np.sum(row_indicators)
  return exact_match_count/n

def one_zero_loss(y_true, y_pred):
  """
  1 - emr
  """
  n = len(y_true)
  row_indicators = np.logical_not(np.all(y_true == y_pred, axis = 1)) # axis = 1 will check for equality along rows.
  not_equal_count = np.sum(row_indicators)
  return not_equal_count/n


def hamming_loss(y_true, y_pred):
    """
	XOR TT for reference - 
	
	A  B   Output
	
	0  0    0
	0  1    1
	1  0    1 
	1  1    0

  Hamming Loss computes the proportion of incorrectly predicted labels to the total number of labels.
	"""
    hl_num = np.sum(np.logical_xor(y_true, y_pred))
    hl_den = np.prod(y_true.shape)
    
    return hl_num/hl_den


def example_based_precision(y_true, y_pred):
    """
    precision = TP/ (TP + FP)
    Example-based precision is defined as the proportion of predicted correct labels to the total number of predicted labels, averaged over all instances
    """
    
    # Compute True Positive 
    precision_num = np.sum(np.logical_and(y_true, y_pred), axis = 1)
    
    # Total number of pred true labels
    precision_den = np.sum(y_pred, axis = 1)
    
    # precision averaged over all training examples
    avg_precision = np.mean(precision_num/precision_den)
    
    return avg_precision

# Label Base Metrics
# ------------------------ Label Based Metrics ------------------------------
# As opposed to example-based metrics, Label based metrics evaluate each label separately and then averaged over all labels.



def label_based_macro_accuracy(y_true, y_pred):
	
    # axis = 0 computes true positives along columns i.e labels
    l_acc_num = np.sum(np.logical_and(y_true, y_pred), axis = 0)

    # axis = 0 computes true postive + false positive + false negatives along columns i.e labels
    l_acc_den = np.sum(np.logical_or(y_true, y_pred), axis = 0)

    # compute mean accuracy across labels. 
    return np.mean(l_acc_num/l_acc_den)


def label_based_macro_precision(y_true, y_pred):
	
	# axis = 0 computes true positive along columns i.e labels
	l_prec_num = np.sum(np.logical_and(y_true, y_pred), axis = 0)

	# axis = computes true_positive + false positive along columns i.e labels
	l_prec_den = np.sum(y_pred, axis = 0)

	# compute precision per class/label
	l_prec_per_class = l_prec_num/l_prec_den

	# macro precision = average of precsion across labels. 
	l_prec = np.mean(l_prec_per_class)
	return l_prec


def label_based_macro_recall(y_true, y_pred):
    
    # compute true positive along axis = 0 i.e labels
    l_recall_num = np.sum(np.logical_and(y_true, y_pred), axis = 0)

    # compute true positive + false negatives along axis = 0 i.e columns
    l_recall_den = np.sum(y_true, axis = 0)

    # compute recall per class/label
    l_recall_per_class = l_recall_num/l_recall_den

    # compute macro averaged recall i.e recall averaged across labels. 
    l_recall = np.mean(l_recall_per_class)
    return l_recall


def label_based_micro_accuracy(y_true, y_pred):
    
    # sum of all true positives across all examples and labels 
    l_acc_num = np.sum(np.logical_and(y_true, y_pred))

    # sum of all tp+fp+fn across all examples and labels.
    l_acc_den = np.sum(np.logical_or(y_true, y_pred))

    # compute mirco averaged accuracy
    return l_acc_num/l_acc_den


def label_based_micro_precision(y_true, y_pred):
    
    # compute sum of true positives (tp) across training examples
    # and labels. 
    l_prec_num = np.sum(np.logical_and(y_true, y_pred))

    # compute the sum of tp + fp across training examples and labels
    l_prec_den = np.sum(y_pred)

    # compute micro-averaged precision
    return l_prec_num/l_prec_den


# Function for Computing Label Based Micro Averaged Recall 
# for a MultiLabel Classification problem. 

def label_based_micro_recall(y_true, y_pred):
	
    # compute sum of true positives across training examples and labels.
    l_recall_num = np.sum(np.logical_and(y_true, y_pred))
    # compute sum of tp + fn across training examples and labels
    l_recall_den = np.sum(y_true)

    # compute mirco-average recall
    return l_recall_num/l_recall_den


def alpha_evaluation_score(y_true, y_pred):
    alpha = 1
    beta = 0.25
    gamma = 1
    
    # compute true positives across training examples and labels
    tp = np.sum(np.logical_and(y_true, y_pred))
    
    # compute false negatives (Missed Labels) across training examples and labels
    fn = np.sum(np.logical_and(y_true, np.logical_not(y_pred)))
    
    # compute False Positive across training examples and labels.
    fp = np.sum(np.logical_and(np.logical_not(y_true), y_pred))
        
    # Compute alpha evaluation score
    alpha_score = (1 - ((beta * fn + gamma * fp ) / (tp +fn + fp + 0.00001)))**alpha 
    
    return alpha_score