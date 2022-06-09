import os
def mount_drive_wandb():

  # Mounting Google Drive
  exec('from google.colab import drive')
  exec("drive.mount('/content/drive')")

  # subprocess.run(['cp', ''])
  os.system('cp ./drive/MyDrive/.gitconfig /root/.gitconfig')
  os.system('cp ./drive/MyDrive/.netrc /root/.netrc')
  os.system('cp ./drive/MyDrive/kaggle.json /root/.kaggle/kaggle.json')
