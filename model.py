"""Model definition for the ConvNeXt model."""

import collections
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from pytorch_lightning import LightningModule
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    """The ConvNeXt block."""

    def __init__(self, l_in, l_out, l_botneck, l_stride=1, l_kern=7) -> None:
        """Initalise the ConvNeXt block."""
        assert l_in == l_out, "input and output channels must be the same"
        super().__init__()

        self.act_fn = nn.GELU()

        # conv. layers:
        # grouped convolution layer first, followed by 2 1x1 layers
        self.conv_b1 = nn.Conv2d(
            l_in,
            l_in,
            kernel_size=(l_kern, 1),
            stride=(l_stride, 1),
            padding=((l_kern - 1) // 2, 0),
            groups=l_in,  # depth-wise convolution => grouped convolution with n_groups = n_channels
        )
        self.norm_b1 = LayerNorm2d(l_in)
        self.conv_b2 = nn.Conv2d(l_in, l_botneck, kernel_size=(1, 1), stride=(1, 1))
        self.conv_b3 = nn.Conv2d(l_botneck, l_out, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        y = self.conv_b1(x)
        y = self.norm_b1(y)
        y = self.conv_b2(y)
        y = self.act_fn(y)
        y = self.conv_b3(y)

        y += x
        return y


class ConvNeXt(LightningModule):
    """ConvNeXt Model."""

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), weight_decay=1e-2)
        lr_scheduler = CustomLRScheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


    def __init__(
        self,
        arch_dt,
        hw_head=(4, 4),
        l_stem=8,
        kernel_size=7,
        samples_per_class=None,
        **kwargs,
    ):
        """Initialise the Model.

        Keyword Arguments:
        -----------------
            arch_dt         -- architecture in dictionary form (required)
                               - n_blocks: number of blocks
                               - n_channels: number of channels for each block
                               - n_botneck: number of channels for bottleneck
            hw_head         -- tuple of the dimension at the input to the head (default (4, 4))

        """
        super().__init__()

        # Calculate pos_weight if samples_per_class is provided
        self.pos_weight = None
        if samples_per_class is not None:
            print(f"------using class frequency weights: {samples_per_class=}")
            weights = samples_per_class.sum() / (
                len(samples_per_class) * samples_per_class
            )
            weights = torch.tensor(weights, dtype=torch.float32)
            self.pos_weight = weights[1] / weights[0]

        # 1. stem:
        self.stem = self._stem_block(l_in=1, l_feats=l_stem)

        # 2. body, consisting of multiple ResNet modular-blocks:
        l_in = l_stem
        mblocks = []
        # iterate over stages:
        for n, n_blocks in enumerate(arch_dt["n_blocks"]):
            stage = self._net_stage(
                n_res_blocks=n_blocks,
                n_stage=n,
                l_in=l_in,
                l_out=arch_dt["n_channels"][n],
                l_botneck=arch_dt["n_botneck"][n],
                l_kern=kernel_size,
            )
            mblocks.append(stage)
            l_in = arch_dt["n_channels"][n]

        self.body = nn.Sequential(
            collections.OrderedDict(
                [("stage_" + str(n + 1), item) for n, item in enumerate(mblocks)],
            ),
        )

        # 3. head:
        head = [nn.AvgPool2d(hw_head)]
        head.extend(
            (
                LayerNorm2d(arch_dt["n_channels"][-1]),
                nn.Flatten(),
                nn.Linear(arch_dt["n_channels"][-1], 1),
            ),
        )
        self.head = nn.Sequential(
            collections.OrderedDict(
                [("head_" + str(n + 1), item) for n, item in enumerate(head)],
            ),
        )

    def _stem_block(self, l_in=1, l_feats=64):
        """Kernel size is equal to stride size to generate a 'patchify' stem."""
        # patchify stem:
        l_kern, l_stride = 4, 4

        stem = [
            nn.Conv2d(
                l_in,
                l_feats,
                kernel_size=(l_kern, 1),
                stride=(l_stride, 1),
                padding=((l_kern - 1) // 2, 0),
            ),
        ]
        stem.append(LayerNorm2d(l_feats))

        return nn.Sequential(
            collections.OrderedDict(
                [("stem_" + str(n + 1), item) for n, item in enumerate(stem)],
            ),
        )

    def _net_stage(self, n_res_blocks, n_stage, **kwargs):
        """Combine ConvNeXt blocks to make stage.

        Keyword Arguments:
        -----------------
        n_res_blocks -- number of blocks within stage
        n_stage      -- current stage number
        kwargs       -- dictionary:
                        - l_in: number of input channels
                        - l_out: number of output channels
                        - l_botneck: number of bottleneck channels
                        - l_kern: length of conv. kernel
                        - n_groups: groups if using grouped-convolution, i.e. ResNeXt

        """
        block_args = deepcopy(kwargs)

        # downsampling block:
        if n_stage != 0:
            all_blocks = [DownsampleLayer(block_args["l_in"], block_args["l_out"])]
            block_args["l_in"] = block_args["l_out"]
        else:
            all_blocks = []

        # iterate over the blocks:
        for _ in range(n_res_blocks):
            all_blocks.append(ConvNeXtBlock(**block_args))
            block_args["l_in"] = block_args["l_out"]

        return nn.Sequential(*all_blocks)

    def activate_logits(self, logits):
        if len(logits.shape) == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)[:, 1]

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.head(x)
        return x.squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.unsqueeze(1)
        y = y.squeeze(0)
        logits = self(x)

        # Convert labels if needed
        labels_tf = y.float()
        if labels_tf.dim() != logits.dim():
            labels_tf = labels_tf.unsqueeze(1)

        loss = F.binary_cross_entropy_with_logits(
            input=logits,
            target=labels_tf,
            pos_weight=self.pos_weight,
        )
        return loss

    def predict_step(self, batch, batch_idx=None):
        x, y = batch
        x = x.unsqueeze(1)
        logits = self(x)
        y_hat = self.activate_logits(logits)
        if y is not None:
            y = y.squeeze(0)

        return y_hat, y

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels of '2D' spatial NCHW tensors."""

    def __init__(self, num_channels, eps=1e-6, affine=True) -> None:
        """Initialise the LayerNorm."""
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            nn.functional.layer_norm(
                x.permute(0, 2, 3, 1).contiguous(),
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )

class DownsampleLayer(nn.Module):
    """Separate layer for downsampling."""

    def __init__(self, l_in, l_out) -> None:
        """Initalise the downsample layer."""
        super().__init__()
        self.lnorm = LayerNorm2d(l_in)
        self.conv = nn.Conv2d(l_in, l_out, kernel_size=(2, 1), stride=(2, 1))

    def forward(self, x):
        x = self.lnorm(x)
        return self.conv(x)


class CustomLRScheduler(LRScheduler):
    """Custom LR Scheduler."""

    def __init__(
        self,
        optimizer,
        start_lr=3e-6,
        max_lr=3e-4,
        freeze_lr=1e-5,
        warmup_epochs=8,
        cool_down_start=50,
        freeze=120,
        last_epoch=-1,
        linear=False,
    ) -> None:
        """Initialise the Custom LR Scheduler."""
        self.start_lr = start_lr
        self.max_lr = max_lr
        self.freeze_lr = freeze_lr
        self.warmup_epochs = warmup_epochs
        self.cooldown_start = cool_down_start
        self.freeze = freeze
        self.linear = linear
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            if self.linear:
                return [
                    self.start_lr
                    + (self.max_lr - self.start_lr)
                    * (self.last_epoch / self.warmup_epochs)
                    for _ in self.base_lrs
                ]
            # log.:
            lr = [
                np.log10(self.start_lr)
                + (np.log10(self.max_lr / self.start_lr))
                * (self.last_epoch / self.warmup_epochs)
                for _ in self.base_lrs
            ]
            return [10 ** (item) for item in lr]
        if self.last_epoch <= self.cooldown_start:
            return [self.max_lr for _ in self.base_lrs]
        if self.last_epoch <= self.freeze:
            # Linear decay
            total_cooldown_epochs = self.freeze - self.cooldown_start
            elapsed_cooldown_epochs = self.last_epoch - self.cooldown_start
            if self.linear:
                lr_decay = (self.max_lr - self.freeze_lr) * (
                    elapsed_cooldown_epochs / total_cooldown_epochs
                )
                return [self.max_lr - lr_decay for _ in self.base_lrs]
            # log:
            lr_decay = (np.log10(self.max_lr / self.freeze_lr)) * (
                elapsed_cooldown_epochs / total_cooldown_epochs
            )
            return [10 ** (np.log10(self.max_lr) - lr_decay) for _ in self.base_lrs]
        return [self.freeze_lr for _ in self.base_lrs]


# -------------------------------------------------------------------
#  some predefined instances of the network
# -------------------------------------------------------------------


class CNXBase(ConvNeXt):
    """base version of the ConvNeXt to vary number of blocks.

    assuming a (1024 x 1) input
    """

    def __init__(self, d=1, w=1, hw_head=(32, 1), **kwargs) -> None:
        """Initialise a CNX Base."""
        base_block_width = [6, 12, 24, 48]
        base_block_ratio = [1, 1, 3, 1]
        n_blocks = [d * i for i in base_block_ratio]
        n_channels = [w * i for i in base_block_width]
        n_botneck = [w * i * 4 for i in base_block_width]
        self.depth = n_blocks
        self.width = n_channels
        super().__init__(
            arch_dt={
                "n_blocks": n_blocks,
                "n_channels": n_channels,
                "n_botneck": n_botneck
            },
            hw_head=hw_head,
            l_stem=n_channels[0],
            **kwargs,
        )


class CNXNano(CNXBase):
    def __init__(self, **kwargs):
        super().__init__(d=1, w=1, **kwargs)

class CNXSmall(CNXBase):
    def __init__(self, **kwargs):
        super().__init__(d=2, w=2, **kwargs)

class CNXMedium(CNXBase):
    def __init__(self, **kwargs):
        super().__init__(d=3, w=4, **kwargs)

class CNXLarge(CNXBase):
    def __init__(self, **kwargs):
        super().__init__(d=3, w=8, **kwargs)

class CNXXLarge(CNXBase):
    def __init__(self, **kwargs):
        super().__init__(d=6, w=10, **kwargs)
