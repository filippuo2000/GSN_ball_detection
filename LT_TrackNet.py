from TrackNet import TrackNet
from TrackNetClassifier import TrackNetClassifier
from TrackNetDataModule import TrackNetDataModule

import pytorch_lightning as pl
from pytorch_lightning import Trainer
import multiprocessing


def main():
    model = TrackNet()
    classifier = TrackNetClassifier(model)
    dm = TrackNetDataModule()
    dm.prepare_data()
    dm.setup()

    trainer = pl.Trainer()
    trainer.fit(model=classifier, datamodule=dm)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()


