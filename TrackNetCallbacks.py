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
    def on_validation_epoch_end(self, trainer, pl_module):
        # Set model to evaluation mode
        pl_module.eval()
        # Get validation dataloader
        val_dataloader = trainer.datamodule.val_dataloader()
        # Get Wandb logger
        wandb_logger = trainer.logger.experiment if isinstance(trainer.logger, WandbLogger) else None

        output_images = []
        labels = []

        # Iterate over batches in the validation set
        with torch.no_grad():
            for batch in val_dataloader:
                if len(output_images) >= self.num_samples:
                    break

                # Forward pass to get model output
                input_img, label = batch
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
                    labels.append((label[0][i].item(), label[1][i].item()))

            # Log the image using wandb
            if wandb_logger:
                wandb_logger.log({"example_val_output_images": [wandb.Image(output_image, caption=y_label) for output_image, y_label in zip(output_images, labels)]})

        # Set model back to training mode
        pl_module.train()