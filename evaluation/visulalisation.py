import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_cf_matrix_predictions(y_true, y_pred):
  """"
  Plots a confusion matrix directly from labels and predictions
  """
  cf_matrix = confusion_matrix(y_true,y_pred)
  group_names = ['True Neg','False Pos','False Neg','True Pos']
  group_counts = ['{0:0.0f}'.format(value) for value in
                  cf_matrix.flatten()]
  group_percentages = ['{0:.2%}'.format(value) for value in
                      cf_matrix.flatten()/np.sum(cf_matrix)]
  labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
  labels = np.asarray(labels).reshape(2,2)
  sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

def plot_cf_matrix(cf_matrix):
  """
  Plots the confusion matrix provided
  """
  group_names = ['True Neg','False Pos','False Neg','True Pos']
  group_counts = ['{0:0.0f}'.format(value) for value in
                  cf_matrix.flatten()]
  group_percentages = ['{0:.2%}'.format(value) for value in
                      cf_matrix.flatten()/np.sum(cf_matrix)]
  labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
  labels = np.asarray(labels).reshape(2,2)
  sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
  
import warnings
import statsmodels.api as sm
import scipy.stats as stats
def visualise_continous(df, columns):
  warnings.filterwarnings('ignore')
  fig,ax = plt.subplots(15,3,figsize=(30,90))
  for index,i in enumerate(columns):
      sns.distplot(df[i],ax=ax[index,0])
      sns.boxplot(df[i],ax=ax[index,1])
      stats.probplot(df[i],plot=ax[index,2])

  fig.tight_layout()
  fig.subplots_adjust(top=0.95)
  plt.suptitle("Visualizing Continuous Columns",fontsize=50)
  plt.show()
