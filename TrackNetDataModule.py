import os
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from InitializeDataset import InitializeDataset

class TrackNetDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 2
        self.data_dir = 'Dataset-small'

    def prepare_data(self):
        self.dataset = InitializeDataset(self.data_dir)

        if os.path.exists("split.txt"):
            self.dataset.read_split("split.txt")
        else:
            raise RuntimeError("No split.txt file - File defining training, val, test division")
        self.dataset.read_labels()
        self.dataset.stats()

    def setup(self, stage=None):
        self.train_dataset = self.dataset.train_dataset()
        self.validate_dataset = self.dataset.validate_dataset()
        self.test_dataset = self.dataset.test_dataset()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers = 4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.validate_dataset, batch_size=self.batch_size, persistent_workers=True, num_workers=1)