from TrackNet import TrackNet
from TrackNetClassifier import TrackNetClassifier
from TrackNetDataModule import TrackNetDataModule
from TrackNetCallbacks import get_early_stopping, get_checkpoint_callback, ImagePredictionLogger
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import wandb
import multiprocessing
import torch.nn as nn


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.uniform_(m.weight, a=-0.05, b=0.05)
            nn.init.uniform_(m.bias, a=-0.05, b=0.05)

def main():
    wandb.login(key="d2e53441a811808da2ad7a72b2b8c36c99d4afaf", verify=True)
    wandb_logger = WandbLogger(project='TrackNetModel', job_type='train')
    wandb.init(reinit=True)
    model = TrackNet()
    initialize_weights(model)
    classifier = TrackNetClassifier(model)
    dm = TrackNetDataModule()
    dm.prepare_data()
    dm.setup()

    trainer = pl.Trainer(check_val_every_n_epoch=1, num_sanity_val_steps=1, accelerator="auto", max_epochs=50,
                         callbacks=[get_checkpoint_callback(), get_early_stopping(), ImagePredictionLogger()],
                         logger=wandb_logger)

    #trainer = pl.Trainer(num_sanity_val_steps=0)
    trainer.fit(model=classifier, datamodule=dm)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()


