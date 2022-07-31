import time
class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
                                                                      
class CustomDataset(data.Dataset):

  def __init__(self, base_dir,):
    super().__init__()
    self.base_dir = base_dir
    self.img_names = os.listdir(base_dir)
    self.transform = T.Compose([
      # T.PILToTensor(),
                                T.Resize((284,284)),
                                T.RandomAutocontrast(0.2), 
                                T.RandomAffine(
                                    degrees = 15,
                                    translate =  (.09,.09),
                                    scale = (.86, 1.20),
                                    shear =  9
                                ),
                                AddGaussianNoise(0,0.013)
                                                      ])
  def __len__(self):
    return len(self.img_names)

  def __getitem__(self, idx):
    img_name =  self.img_names[idx]
    img_path = self.base_dir + img_name
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = np.expand_dims(img,0)
    try:
      img = torch.from_numpy(img)
    except TypeError as e:
      print('One Error')
      img = torch.ones((1,284,284))
      return img, img_name
    img = self.transform(img)
    return img,img_name

#  This is the point where the loop starts and you can either batch you datset if you have it whole
# or do it in batches if you need to incrementally download it
for base in range(54,55):
  base = str(base)
  if not os.path.exists(f'{base}.zip'):
    # time.sleep(60)\
    break
    # continue
  print(f'Working on {base} number')
  os.makedirs(base + '_img', exist_ok = True)
  !unzip -q {base}.zip -d {base}_img
  print('Now Working on Dataset')
  train_ds = CustomDataset(f'{base}_img/')
  train_dataloader = data.DataLoader(train_ds, batch_size = len(train_ds) )
  try:
    img, label = next(iter(train_dataloader))
    with h5py.File(f'{base}.h5', 'w') as f:
      f.create_dataset('img', data = img.numpy())
      f.create_dataset('name', data = label)
    shutil.copy(f'{base}.h5', f'./drive/MyDrive/padchest/{base}.h5')
  except TypeError as e:
    print(f'Error in Batch {base}')
    raise e
  # print("this ran too")
  os.remove(f'./{base}.h5')
  os.remove(f'./{base}.zip')
  shutil.rmtree(f'./{base}_img')
