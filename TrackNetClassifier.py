import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import cv2
import numpy as np
import wandb
from TrackNetMetrics import MyAccuracy
from pytorch_lightning.callbacks import Callback



class TrackNetClassifier(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.lr = 1
        self.size = (360, 640)
        self.sigma = 10
        self.accuracy = MyAccuracy()

    def forward(self, x):
        return self.model(x)

    def gaussian_distribution(self, xy):
        device = torch.device('cuda') if xy[0].is_cuda else torch.device('cpu')

        batch_size = xy[0].size(0)
        xx, yy = torch.meshgrid(torch.arange(self.size[0], device=device), torch.arange(self.size[1], device=device))
        xx = xx.unsqueeze(0).expand(batch_size, -1, -1)  # Expand dimensions for broadcasting
        yy = yy.unsqueeze(0).expand(batch_size, -1, -1)

        xy_batch = [item.unsqueeze(1).to(device) / 2 for item in xy]

        # Reshape tensors to ensure compatible sizes for concatenation
        xy_batch[0] = xy_batch[0].reshape(batch_size, 1, 1, -1)
        xy_batch[1] = xy_batch[1].reshape(batch_size, 1, 1, -1)

        xy_batch = torch.cat(xy_batch, dim=3)  # Correct concatenation

        distances = ((xx - xy_batch[:, :, :, 0]) ** 2 + (yy - xy_batch[:, :, :, 1]) ** 2) / (2 * self.sigma ** 2)

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
        output_y = torch.zeros((batch_size, 256, self.size[0], self.size[1]), device=device)
        index_tensor = torch.arange(256, device=device).view(1, -1, 1, 1)
        index_tensor = index_tensor.expand(batch_size, -1, -1, -1)
        mask = index_tensor == y.unsqueeze(1)
        mask = mask.to(device)
        output_y[mask] = 1
        return output_y

    def postprocess_output(self, feature_map: torch.Tensor, scale=2):
        # expects a feature map of size [B, 256, 360, 640] - this is the network's output
        # to match the shape of the label values - List[torch.Tensor, torch.Tensor] - [2, B]
        feature_map, _ = torch.max(feature_map, dim=1)
        feature_map = feature_map.cpu()
        locations = torch.zeros((2, feature_map.shape[0])) - 1
        feature_map = torch.transpose(feature_map, 1, 2)
        feature_map = feature_map.detach().numpy()
        feature_map *= 255
        feature_map = feature_map.astype(np.uint8)
        #print("feature map shape is: ", feature_map.shape)
        ret, heatmap = cv2.threshold(feature_map[:], 127, 255, cv2.THRESH_BINARY)
        #print("heatmap shape is: ", heatmap.shape)
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0, 0)
        #print("heatmap shape is: ", heatmap.shape)
        # show_img(np.expand_dims(heatmap, axis=0))
        for i in range(feature_map.shape[0]):
            circles = cv2.HoughCircles(heatmap[i], cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=100, param2=2,
                                       minRadius=2,
                                       maxRadius=7)
            x, y = -1, -1
            if circles is not None:
                if len(circles) == 1:
                    x = circles[0][0][1] * scale
                    y = circles[0][0][0] * scale
            locations[0][i], locations[1][i] = x, y

        return locations

    def compute_loss(self, x, y):
        return  F.binary_cross_entropy(x, y)

    def common_step(self, batch, batch_idx):
        x, y = batch
        y_img = self.gaussian_distribution(y)
        y_img = self.y_3d(y_img)
        outputs = self(x)
        loss = self.compute_loss(outputs, y_img)
        return loss, outputs, y

    def common_test_valid_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, batch_idx)
        preds = self.postprocess_output(outputs)
        preds = preds.device(self.device)
        acc = self.accuracy(preds, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        # self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        return loss
        #accuracy = self.accuracy()
        #_, acc = self.common_test_valid_step(batch, batch_idx)
        # self.log_dict(  # I sposób logowania
        #     {
        #         "train_loss": loss,
        #         #"train_accuracy": acc
        #     },
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True
        # )
        # return {'loss':loss}

    def validation_step(self, batch, batch_idx):
        loss, acc = self.common_test_valid_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss


    def test_step(self, batch, batch_idx):
        loss, acc = self.common_test_valid_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss


    def configure_optimizers(self):
        #return torch.optim.Adam(self.parameters(), lr = 1e-1)
        return torch.optim.Adadelta(self.parameters(), lr=self.lr)


class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=8):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds, _ = torch.max(logits, dim=1)

        # Log the images as wandb Image
        trainer.logger.experiment.log({
        "examples":[wandb.Image(x, caption=f"random, Label:{y}")
                       for x, y in zip(val_imgs[:self.num_samples],
                                             val_labels[:self.num_samples])]
                       })