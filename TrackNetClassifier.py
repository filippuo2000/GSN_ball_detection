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

    def gaussian_distribution(self, xy, sigma, size):
        device = torch.device('cuda') if xy[0].is_cuda else torch.device('cpu')

        batch_size = xy[0].size(0)
        xx, yy = torch.meshgrid(torch.arange(size[0], device=device), torch.arange(size[1], device=device))
        xx = xx.unsqueeze(0).expand(batch_size, -1, -1)  # Expand dimensions for broadcasting
        yy = yy.unsqueeze(0).expand(batch_size, -1, -1)

        xy_batch = [item.unsqueeze(1).to(device) / 2 for item in xy]

        # Reshape tensors to ensure compatible sizes for concatenation
        xy_batch[0] = xy_batch[0].reshape(batch_size, 1, 1, -1)
        xy_batch[1] = xy_batch[1].reshape(batch_size, 1, 1, -1)

        xy_batch = torch.cat(xy_batch, dim=3)  # Correct concatenation

        distances = ((xx - xy_batch[:, :, :, 0]) ** 2 + (yy - xy_batch[:, :, :, 1]) ** 2) / (2 * sigma ** 2)

        G = torch.exp(-distances) * 255
        G = torch.floor(G)

        # Handle cases where either x or y is empty
        empty_mask = (xy[0] == -1) | (xy[1] == -1)
        empty_mask = empty_mask.to(device)
        G[empty_mask] = 0
        return G.int()

    def y_3d(self, y):
        device = torch.device('cuda') if y.is_cuda else torch.device('cpu')

        batch_size = y.size(0)
        output_y = torch.zeros((batch_size, 256, 360, 640), device=device)
        index_tensor = torch.arange(256, device=device).view(1, -1, 1, 1)
        index_tensor = index_tensor.expand(batch_size, -1, -1, -1)
        mask = index_tensor == y.unsqueeze(1)
        mask = mask.to(device)
        output_y[mask] = 1
        return output_y

    def compute_loss(self, x, y):
        return  F.binary_cross_entropy(x, y)

        #return F.binary_cross_entropy(x, y)


    def common_step(self, batch, batch_idx):
        x, y = batch
        #print(y[1].shape)
        y = self.gaussian_distribution(y, sigma=10, size=(360, 640))
        y = self.y_3d(y)
        #print(y[1])
        #print(x.shape)
        #print(y)
        #print(type(y))
        #print(y.shape)
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
        self.log_dict(  # I sposób logowania
            {
                "train_loss": loss,
                #"train_accuracy": accuracy
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_test_valid_step(batch, batch_idx)
        #self.validation_step_output

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        #return torch.optim.Adam(self.parameters(), lr = 1e-1)
        return torch.optim.Adadelta(self.parameters(), lr=1)