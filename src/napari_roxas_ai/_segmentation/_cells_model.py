"""
Cells Segmentation Model Architecture
"""

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import skimage.measure as skm
import torch
import torchvision.transforms as transforms


class CellsSegmentationModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.available_device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.mps.is_available() else "cpu"
        )
        self.use_autocast = torch.amp.autocast_mode.is_autocast_available(
            self.available_device
        )
        self.net = smp.Unet(
            encoder_weights=None, classes=2, encoder_name="resnet50"
        ).to(device=self.available_device)
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn("resnet50")
        self.eval()

    def load_weights(self, path):
        self.load_state_dict(
            torch.load(
                path,
                weights_only=True,
                map_location=self.available_device,
            )["state_dict"]
        )

    def forward(self, x):
        out = self.net(x)
        out = torch.argmax(out, dim=1, keepdim=True).long()
        return out

    def infer(self, img):

        # Patch size and stride
        patch_size = 1024
        stride = 800
        pad_size = int(
            (patch_size - stride) / 2
        )  # Warning: (patch_size - stride) / 2 must be an int
        pad_mode = "reflect"

        # Pad the image in reflect mode to avoid edge effects
        img_padded = np.pad(
            img,
            (
                (pad_size, pad_size),
                (pad_size, pad_size),
                (0, 0),
            ),
            mode=pad_mode,
        )
        labels_padded = np.zeros((img_padded.shape[0], img_padded.shape[1]))

        # Process the image in chunks
        for i in range(img_padded.shape[0] // stride + 1):
            for j in range(img_padded.shape[1] // stride + 1):
                x, dx, y, dy = (
                    stride * i,
                    stride * i + patch_size,
                    stride * j,
                    stride * j + patch_size,
                )
                patch = img_padded[x:dx, y:dy].copy()

                # Pad to patch size if smaller
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    patch = np.pad(
                        patch,
                        (
                            (0, max(0, patch_size - patch.shape[0])),
                            (0, max(0, patch_size - patch.shape[1])),
                            (0, 0),
                        ),
                        mode=pad_mode,
                    )

                patch = self.preprocessing_fn(patch)
                patch = transforms.ToTensor()(patch).float()
                patch = torch.unsqueeze(patch, 0)
                patch = patch.to(device=self.available_device)
                with torch.no_grad():
                    if self.use_autocast:
                        with torch.autocast(device_type=self.available_device):
                            pred = self(patch)
                    else:
                        pred = self(patch)
                pred = torch.squeeze(pred.cpu().detach()).long().numpy()
                patch_labels = skm.label(pred).transpose((1, 0))
                patch_labels = (patch_labels.T > 0).astype(int) * 255

                update_zone_shape = labels_padded[
                    x + pad_size : dx - pad_size, y + pad_size : dy - pad_size
                ].shape
                labels_padded[
                    x + pad_size : dx - pad_size, y + pad_size : dy - pad_size
                ] = patch_labels[
                    pad_size : update_zone_shape[0] + pad_size,
                    pad_size : update_zone_shape[1] + pad_size,
                ]

        return labels_padded[pad_size:-pad_size, pad_size:-pad_size]
