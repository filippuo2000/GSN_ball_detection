import os
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from InitializeDataset import InitializeDataset
from CustomTransforms import ResizeToCustomSize
from CustomTransforms import GaussianDistributionTransform
from CustomTransforms import ToFloatTensor



class TrackNetDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 2
        self.data_dir = 'Dataset'

    def prepare_data(self):
        self.dataset = InitializeDataset("Dataset")

        if os.path.exists("split.txt"):
            self.dataset.read_split("split.txt")
        else:
            self.dataset.initialize_data()
            self.dataset.random_split((500, 1))
        self.dataset.read_labels()
        self.dataset.stats()

    def setup(self, stage=None):
        transform = transforms.Compose([
            ToFloatTensor(),
            ResizeToCustomSize(360, 640),
        ]
        )

        transform_label = transforms.Compose([
            GaussianDistributionTransform(sigma=10, size=(360, 640)),
        ])

        self.train_dataset = self.dataset.train_dataset()
        self.validate_dataset = self.dataset.validate_dataset()
        self.test_dataset = self.dataset.test_dataset()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers = 4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.validate_dataset, batch_size=self.batch_size, persistent_workers=True, num_workers=1)