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
            conv_layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv_filters[i],
                kernel_size=tuple(conv_kernels[i]),
                stride=tuple(conv_strides[i])
            ))

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

            in_channels = conv_filters[i]

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
        print('x:',x.shape)
        x = self.conv_stack(x)
        print('x-conv: ',x.shape)
        x = torch.flatten(x, 1)
        print('x-flat: ',x.shape)
        return self.fc_stack(x)