from TrackNet import TrackNet
from TrackNetClassifier import TrackNetClassifier
from TrackNetDataModule import TrackNetDataModule
from pytorch_lightning.loggers import TensorBoardLogger

import pytorch_lightning as pl
from pytorch_lightning import Trainer
import multiprocessing


def main():
    logger = TensorBoardLogger("lightning_logs", name="GSN_ball_detection")
    model = TrackNet()
    classifier = TrackNetClassifier(model)
    dm = TrackNetDataModule()
    dm.prepare_data()
    dm.setup()

    trainer = pl.Trainer(num_sanity_val_steps=3, logger=logger)
    trainer.fit(model=classifier, datamodule=dm)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()


