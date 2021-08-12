from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from models.litmodel import LitModel
from models.fcn32s import FCN32s


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torchvision.models.vgg import VGG
import pytorch_lightning as pl
from torch.optim.rmsprop import RMSprop


from dataloader import TrainDataset, lit_custom_data
from pytorch_lightning import loggers
from configs import Configs
from pytorch_lightning.callbacks import Callback
# os.environ["OPENBLAS_MAIN_FREE"] = '1'


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-epoch-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_epoch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if self.use_modelcheckpoint_filename:
            filename = trainer.checkpoint_callback.filename
        else:
            filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
        ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
        trainer.save_checkpoint(ckpt_path)
            

pl.utilities.seed.seed_everything(0)
if __name__ == '__main__':
    hparams = {
        'lr': 0.0019054607179632484
    }
    model = LitModel(hparams)
    
    # model = LitModel.load_from_checkpoint("./lightning_logs/version_24/checkpoints/N-epoch-Checkpoint_epoch=19_global_step=9199.ckpt")

    dataset = lit_custom_data()
    dataset.setup()
    trainer = pl.Trainer(gpus=1, max_epochs=120, callbacks=[CheckpointEveryNSteps(1)])
    trainer.fit(model, dataset)
    print("hello")
