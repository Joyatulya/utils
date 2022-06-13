from google.colab import drive
import os

def mount_drive():
  print('Mounting Drive')
  drive.mount('/gdrive')

def config_files():
  print('Copying Kaggle/wandb/git')
  os.system('cp ./drive/MyDrive/kaggle.json /root/.kaggle/kaggle.json')
  os.system('cp ./drive/MyDrive/.gitconfig /root/.gitconfig')
  os.system('cp ./drive/MyDrive/.netrc /root/.netrc')