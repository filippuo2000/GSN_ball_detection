import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import cv2
import numpy as np
from TrackNetMetrics import MyMetrics
from CrossEntropy import CrossEntropy

class TrackNetClassifier(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.lr = 1
        self.size = (360, 640)
        self.sigma = 3.3
        self.metrics = MyMetrics()
        self.cross_entropy = CrossEntropy(epsilon=1e-10)
    def forward(self, x):
        return self.model(x)

    def gaussian_distribution(self, xy):
        # expects list with two tensors, first tensor consists of x values, second - y values
        # returns feature map, based on given xy coordinates, in shape [B, 360, 640]
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
        distances = ((xx - xy_batch[:, :, :, 1]) ** 2 + (yy - xy_batch[:, :, :, 0]) ** 2) / (2 * self.sigma ** 2)

        G = torch.exp(-distances) * 255
        G = torch.floor(G)
        # Handle cases where either x or y is empty (denoted as value = -100)
        empty_mask = (xy[0] == -100) | (xy[1] == -100)
        empty_mask = empty_mask.to(device)
        G[empty_mask] = 0
        return G.int()
        #return G.long()

    def y_onehot(self, y):
        # expects a feature map of shape [B, 360, 640]
        # returns one hot encoding in shape [B, 256, 360, 640]
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
        _ , feature_map = torch.max(feature_map, dim=1)
        feature_map = feature_map.float()
        feature_map = feature_map.cpu()
        locations = torch.zeros((2, feature_map.shape[0])) - 1
        feature_map = torch.transpose(feature_map, 1, 2)
        feature_map = feature_map.detach().numpy()
        #feature_map *= 255
        feature_map = feature_map.astype(np.uint8)
        #print("feature map shape is: ", feature_map.shape)
        ret, heatmap = cv2.threshold(feature_map[:], 127, 255, cv2.THRESH_BINARY)
        #print("heatmap shape is: ", heatmap.shape)
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0, 0)
        #print("heatmap shape is: ", heatmap.shape)
        # show_img(np.expand_dims(heatmap, axis=0))
        for i in range(feature_map.shape[0]):
            circles = cv2.HoughCircles(heatmap[i], cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=100, param2=0.9,
                                       minRadius=1,
                                       maxRadius=15)
            x, y = -10, -10
            if circles is not None:
                if len(circles) == 1:
                    x = circles[0][0][1] * scale
                    y = circles[0][0][0] * scale
            locations[0][i], locations[1][i] = x, y

        return locations

    def compute_loss(self, x, y):
        return self.cross_entropy(x, y)
        #return  F.binary_cross_entropy(x, y)
        #return F.cross_entropy(x, y)

    def common_step(self, batch, batch_idx):
        x, y = batch
        #prepare y feature map
        y_img = self.gaussian_distribution(y)
        y_img = self.y_onehot(y_img)

        outputs = self(x)
        loss = self.compute_loss(outputs, y_img)
        return loss, outputs, y

    def common_test_valid_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, batch_idx)
        #locate xy coordinates basing on resulting feature map
        preds = self.postprocess_output(outputs)
        preds = preds.to(device=self.device)
        metrics = self.metrics(preds, y)
        acc = metrics['accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        return loss, acc, precision, recall

    def training_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, acc, precision, recall = self.common_test_valid_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        return loss


    def test_step(self, batch, batch_idx):
        loss, acc, precision, recall = self.common_test_valid_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_precision', precision, prog_bar=True)
        self.log('test_recall', recall, prog_bar=True)
        return loss


    def configure_optimizers(self):
        #return torch.optim.Adam(self.parameters(), lr = 1e-1)
        return torch.optim.Adadelta(self.parameters(), lr=self.lr)


