import pytorch_lightning as pl
import torch
from torchvision import models, nn
from torch import optim

class SimCLR(pl.LightningModule):

  def __init__(
      self,
      temp,
      lr,
      weight_decay,
      steps_per_epoch,
      hidden_dim = 128,
      max_epochs= 100,
       ):
    
    super().__init__()
    self.save_hyperparameters()
    self.encoder = models.regnet_x_8gf(weights= None, num_classes = 4 * hidden_dim)
    self.encoder.stem = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(True)
)
    self.classifier = nn.Sequential(
        nn.ReLU(True),
        nn.Linear(hidden_dim * 4, hidden_dim * 4),
        nn.ReLU(True),
        nn.Linear(hidden_dim * 4, hidden_dim)
    )
  def forward(self, x):
    x = self.encoder(x)
    return self.classifier(x)

  def training_step(self, batch, batch_idx):
    return self.info_nce_loss(batch, 'train')

  def validation_step(self, batch, batch_idx):
    return self.info_nce_loss(batch, 'val')
    
  def configure_optimizers(self):
    optimizer = optim.AdamW(
        self.parameters(),
        lr = self.hparams.lr,
        weight_decay = self.hparams.weight_decay
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max = self.hparams.steps_per_epoch * self.hparams.max_epochs,
                                                        eta_min = 1e-5)
    with_warmup = create_lr_scheduler_with_warmup(
        lr_scheduler,
        warmup_start_value = 1e-5,
        warmup_duration = self.hparams.steps_per_epoch * 5,
        warmup_end_value = .8,
        )
    return [optimizer], [lr_scheduler]

  def get_latent(self, x):
    return self.encoder(x)

  def info_nce_loss(self, batch, mode = 'train'):
    xi, xj = batch
    batch_size = xi.shape[0]

    imgs= torch.cat([xi, xj], 0)
    feats = self(imgs)

    cos_sim = F.cosine_similarity(feats[:,None,:] ,feats[None,:,:], dim = -1)
    # Here I have gotten the similarity of each image with each other image
    mask = torch.eye(cos_sim.shape[0], dtype = torch.bool, device = self.device)
    cos_sim.masked_fill_(mask, -9e15)
    cos_sim /= self.hparams.temp
    pos_mask = mask.roll(cos_sim.shape[0] // 2, dims = 0)
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, -1)
    nll = nll.mean()

    #  Logging all the above shit   
    self.log(mode+'_loss', nll)

    comb_sim = torch.cat([cos_sim[pos_mask][:,None], cos_sim.masked_fill(pos_mask, -9e15)], -1)
    sim_argsort = comb_sim.argsort(dim = -1, descending = True).argmin(dim = -1)
    # Logging ranking metrics
    self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean())
    self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean())
    self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())

    return nll
