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
        self.available_device = "cuda" if torch.cuda.is_available() else "cpu"
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
        img_pad = np.zeros(
            (img.shape[0] + 224, img.shape[1] + 224, img.shape[2])
        )
        label_pad = np.zeros((img.shape[0] + 224, img.shape[1] + 224))
        img_pad[112:-112, 112:-112] = img
        for i in range(
            img_pad.shape[0] // 800 + 1
        ):  # split image into 1024*1024 chunks
            for j in range(img_pad.shape[1] // 800 + 1):
                x, x1, y, y1 = 800 * i, 800 * i + 1024, 800 * j, 800 * j + 1024
                to_much_x = x1 - img_pad.shape[0]
                to_much_y = y1 - img_pad.shape[1]
                if to_much_x > 0:
                    x = img_pad.shape[0] - 1024
                    x1 = img_pad.shape[0]
                if to_much_y > 0:
                    y = img_pad.shape[1] - 1024
                    y1 = img_pad.shape[1]
                input_img = img_pad[x:x1, y:y1].copy()
                img1 = self.preprocessing_fn(input_img)
                img1 = transforms.ToTensor()(img1).float()
                img1 = torch.unsqueeze(img1, 0)
                img1 = img1.to(device=self.available_device)
                with torch.no_grad():
                    if self.use_autocast:
                        with torch.autocast(device_type=self.available_device):
                            r = self(img1)
                    else:
                        r = self(img1)
                processed = (
                    torch.squeeze(r.cpu().detach()).long().numpy()
                )  # process to masks
                labels = skm.label(processed).transpose(
                    (1, 0)
                )  # todo why strange transpose
                result_semseg = (labels.T > 0).astype(int) * 255
                label_pad[x + 112 : x1 - 112, y + 112 : y1 - 112] = (
                    result_semseg[112:-112, 112:-112]
                )
        return label_pad[112:-112, 112:-112]
