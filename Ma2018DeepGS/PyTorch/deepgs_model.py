import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DeepGSModel(nn.Module):
    def __init__(self, cnnFrame, markerImage):
        super().__init__()

        img_h, img_w = markerImage

        # ----- Extract CNN parameters -----
        conv_kernels = [list(map(int, k.split("*"))) for k in cnnFrame["conv_kernel"]]
        conv_strides = [list(map(int, s.split("*"))) for s in cnnFrame["conv_stride"]]
        conv_filters = cnnFrame["conv_num_filter"]
        pool_kernels = [list(map(int, k.split("*"))) for k in cnnFrame["pool_kernel"]]
        pool_strides = [list(map(int, s.split("*"))) for s in cnnFrame["pool_stride"]]
        pool_types = cnnFrame["pool_type"]
        act_types = cnnFrame["pool_act_type"]

        # ----- Build Convolutional stack -----
        conv_layers = []
        in_channels = 1
        for i in range(len(conv_filters)):
            out_ch = conv_filters[i]
            conv_layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv_filters[i],
                kernel_size=tuple(conv_kernels[i]),
                stride=tuple(conv_strides[i])
            ))

            if cnnFrame['norm_layer'] == ['GroupNorm']: 
                # *** ADD GROUPNORM HERE ***            
                # # 8 groups is a good default; must divide out_ch            
                num_groups = min(8, out_ch)  # fallback if out_ch < 8            
                conv_layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_ch))


            # Activation
            if act_types[i] == "relu":
                conv_layers.append(nn.ReLU())
            elif act_types[i] == "sigmoid":
                conv_layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unsupported activation: {act_types[i]}")

            # Pooling
            if pool_types[i] == "max":
                conv_layers.append(nn.MaxPool2d(
                    kernel_size=tuple(pool_kernels[i]),
                    stride=tuple(pool_strides[i])
                ))
            else:
                conv_layers.append(nn.AvgPool2d(
                    kernel_size=tuple(pool_kernels[i]),
                    stride=tuple(pool_strides[i])
                ))

            in_channels = out_ch

        self.conv_stack = nn.Sequential(*conv_layers)

        # ----- Dummy forward pass to determine FC input size -----
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_h, img_w)
            dummy_out = self.conv_stack(dummy)
            flattened_size = dummy_out.numel()

        # ----- Fully connected layers -----
        fc_sizes = cnnFrame["fullayer_num_hidden"]
        fc_act = cnnFrame["fullayer_act_type"]
        dropouts = cnnFrame["drop_float"]

        fc_layers = []
        in_features = flattened_size

        for i in range(len(fc_sizes)):
            fc_layers.append(nn.Dropout(dropouts[i]))
            fc_layers.append(nn.Linear(in_features, fc_sizes[i]))

            # Final layer (1 neuron) → no activation
            if i < len(fc_sizes) - 1:
                if fc_act[i] == "relu":
                    fc_layers.append(nn.ReLU())
                elif fc_act[i] == "sigmoid":
                    fc_layers.append(nn.Sigmoid())
                else:
                    raise ValueError("Unsupported FC activation")

            in_features = fc_sizes[i]

        fc_layers.append(nn.Dropout(dropouts[-1]))
        self.fc_stack = nn.Sequential(*fc_layers)

    def forward(self, x):
        # print('x:',x.shape)
        x = self.conv_stack(x)
        # print('x-conv: ',x.shape)
        x = torch.flatten(x, 1)
        # print('x-flat: ',x.shape)
        return self.fc_stack(x)


import torch.nn.functional as F

class LightGSModel(nn.Module):
    """
    A low-capacity, stable, low-overfitting CNN architecture for genomic prediction.
    Designed specifically for genotype matrices (1×H×W) on CPU/MPS/GPU.
    """

    def __init__(self, markerImage):
        super().__init__()

        H, W = markerImage

        # ---- Convolutional block ----
        # Only one convolution layer with small number of filters
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(1, 5),
            stride=(1, 1),
            padding=(0,2),
            bias=True
        )

        # GroupNorm is stable and MPS-friendly
        self.norm = nn.GroupNorm(num_groups=4, num_channels=16)

        # Dropout in convolution helps more than usual for genomic data
        self.dropout_conv = nn.Dropout2d(p=0.2)

        # Max pooling to reduce spatial size and capacity
        self.pool = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))

        # ---- Determine flatten size ----
        with torch.no_grad():
            dummy = torch.zeros(1, 1, H, W)
            x = self.pool(self.dropout_conv(F.relu(self.norm(self.conv(dummy)))))
            flattened = x.numel()

        self.fc1 = nn.Linear(flattened, 32)
        self.dropout_fc = nn.Dropout(p=0.5)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout_conv(x)
        x = self.pool(x)           # (N,16,H, W/2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        return self.fc_out(x)


class LightGS1D(nn.Module):
    """
    Low-capacity 1D CNN for genotype -> phenotype regression.
    Very stable on MPS; reduced overfitting vs DeepGS.
    """
    def __init__(self, num_markers):
        super().__init__()
        C = 32

        # Conv1d over marker dimension
        self.conv = nn.Conv1d(in_channels=1, out_channels=C,
                              kernel_size=9, stride=1, padding=4, bias=True)
        # GroupNorm over channels works for 1D as well
        self.norm = nn.GroupNorm(num_groups=8, num_channels=C)
        self.drop_conv = nn.Dropout(p=0.2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # halve length

        # Determine flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_markers)
            x = self.pool(self.drop_conv(F.relu(self.norm(self.conv(dummy)))))
            flattened = x.numel()

        self.fc1 = nn.Linear(flattened, 32)
        self.drop_fc = nn.Dropout(p=0.5)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x):            # x: (N, 1, M)
        x = self.conv(x)             # (N, C, M)
        x = self.norm(x)
        x = F.relu(x)
        x = self.drop_conv(x)
        x = self.pool(x)             # (N, C, M/2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop_fc(x)
        return self.fc_out(x)