import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class TrackNetClassifier(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.lr = 1

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, x, y):
        return  F.binary_cross_entropy(x, y)


    def common_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.compute_loss(outputs, y)
        return loss, outputs, y

    def common_test_valid_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, batch_idx)
        #preds = torch.argmax(outputs, dim=1)
        #acc = 0
        return loss

    def training_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, batch_idx)
        #accuracy = self.accuracy()
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 1e-1)