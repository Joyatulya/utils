import os

def load_unzip_kaggle_dataset(dataset_name):

  # Mounting Google Drive
  exec('from google.colab import drive')
  exec("drive.mount('/content/drive')")

  # Moving kaggle.json to the root directory
  os.makedirs('/root/.kaggle',exist_ok=True)
  os.system('cp ./drive/MyDrive/kaggle.json /root/.kaggle/')

  # Downloading the dataset
  os.system(f'kaggle datasets download -d {dataset_name}')

  # Unzipping to file system
  os.system(f'{dataset_name}.zip')