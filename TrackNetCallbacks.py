from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb

import cv2
import numpy as np


def get_early_stopping():
    early_stop_callback = EarlyStopping(
       monitor='val_loss',
       patience=3,
       verbose=False,
       mode='min'
    )
    return early_stop_callback

def get_checkpoint_callback():
    MODEL_CKPT_PATH = 'model/'
    MODEL_CKPT = 'model-{epoch:02d}-{val_loss:.2f}'

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=MODEL_CKPT_PATH,
        filename=MODEL_CKPT,
        save_top_k=3,
        mode='min'
    )
    return checkpoint_callback

class ImagePredictionLogger(Callback):
    def __init__(self, num_samples=8):
        super().__init__()
        self.num_samples = num_samples

        self.size = (360, 640)
        self.sigma = 3.3

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
        return G
        #return G.long()

    def on_validation_epoch_end(self, trainer, pl_module):
        # Set model to evaluation mode
        pl_module.eval()
        # Get validation dataloader
        val_dataloader = trainer.datamodule.val_dataloader()
        # Get Wandb logger
        wandb_logger = trainer.logger.experiment if isinstance(trainer.logger, WandbLogger) else None

        output_images = []
        #y_images = []
        labels = []

        # Iterate over batches in the validation set
        with torch.no_grad():
            for batch in val_dataloader:
                if len(output_images) >= self.num_samples:
                    break

                # Forward pass to get model output
                input_img, label = batch
                #y_img = self.gaussian_distribution(label).to(device=pl_module.device)
                input_img = input_img.to(device=pl_module.device)
                output = pl_module(input_img)
                _ , output = torch.max(output, dim=1)
                output = output.float()
                #print (output.shape)

                #output = output.cpu()
                #output = torch.transpose(output, 1, 2)
                #output = output.detach().numpy()
                #output *= 255
                #output = output.astype(np.uint8)
                ## print("feature map shape is: ", feature_map.shape)
                #_, output = cv2.threshold(output[:], 127, 255, cv2.THRESH_BINARY)
                #print(output.shape)

                for i in range(output.shape[0]):
                    output_images.append(output[i])
                    #y_images.append(y_img[i])
                    labels.append((label[0][i].item(), label[1][i].item()))

            # Log the image using wandb
            if wandb_logger:
                wandb_logger.log({"example_val_output_images": [wandb.Image(output_image, caption=y_label) for output_image, y_label in zip(output_images, labels)]})
                #wandb_logger.log({"example_val_y_images": [wandb.Image(y_image, caption=y_label) for y_image, y_label in zip(y_images, labels)]})

        # Set model back to training mode
        pl_module.train()