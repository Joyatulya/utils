

for num_h5, h5 in tqdm(enumerate((HOME_DIR / 'vinbig').glob('*.h5')), total = 17):
  with h5py.File(h5, 'r') as f:
    _imgs = torch.tensor(np.array(f['img']), dtype = torch.float)
    _names = np.array(f['name'])
  # filter_df = vinbig_df[vinbig_df.image_id == _names[0].decode()]
  for image_idx, image_name in tqdm(enumerate(_names),total = len(_names)):
    vinbig_df.loc[vinbig_df.image_id == image_name.decode(), ['file', 'file_index']] = (f'vinbig_{num_h5}', image_idx
