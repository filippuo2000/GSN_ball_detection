from TrackNet import TrackNet
from TrackNetClassifier import TrackNetClassifier, ImagePredictionLogger
from TrackNetDataModule import TrackNetDataModule
from TrackNetCallbacks import get_early_stopping, get_checkpoint_callback
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import wandb
from pytorch_lightning import Trainer
import multiprocessing


def main():
    wandb.login(key="d2e53441a811808da2ad7a72b2b8c36c99d4afaf", verify=True)
    wandb_logger = WandbLogger(project='TrackNetModel', job_type='train')
    wandb.init(reinit=True)
    model = TrackNet()
    classifier = TrackNetClassifier(model)
    dm = TrackNetDataModule()
    dm.prepare_data()
    dm.setup()

    val_samples = next(iter(dm.val_dataloader()))

    trainer = pl.Trainer(check_val_every_n_epoch=5, num_sanity_val_steps=0, accelerator="auto", max_epochs=50,
                         callbacks=[get_checkpoint_callback(), get_early_stopping(), ImagePredictionLogger(val_samples)],
                         logger=wandb_logger)

    trainer = pl.Trainer(num_sanity_val_steps=0)
    trainer.fit(model=classifier, datamodule=dm)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()


