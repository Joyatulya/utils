def get_imgs_from_folder(
  folder,
  img_type : str = 'png',
  img_size = (284,284),
  normalize = (119, 69)
):
  """
  Get all the specified images withing a folder into a tensor
  Params
  ---
  folder : PosixPath -> The folder which has teh images
  img_type = png/jpg/jpeg
  img_size = Dims of the images
  normalize = Through which values should all the images be normalized
  """
  
  imgs = list(folder.glob(f"*.jpeg"))
  imgs_tensor = torch.zeros(len(imgs), 3, *img_size)
  
  transform = T.Compose([
      T.Resize(img_size),
      T.Normalize(normalize)
  ])
  
  for i, img in tqdm(enumerate(imgs), total = len(imgs)):
    img = io.read_image(str(img), ImageReadMode.RGB).float()
    img = transform(img)
    imgs_tensor[i] = img.clone(
