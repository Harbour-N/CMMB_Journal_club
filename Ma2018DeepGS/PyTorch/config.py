cnnFrame = {
    "conv_kernel": ["3*3", "3*3"],
    "conv_stride": ["1*1", "1*1"],
    "conv_num_filter": [32, 64],
    "pool_act_type": ["relu", "relu"],
    "pool_type": ["max", "max"],
    "pool_kernel": ["2*2", "2*2"],
    "pool_stride": ["2*2", "2*2"],
    "drop_float": [0.25, 0.25, 0.5],
    "fullayer_num_hidden": [128, 1],
    "fullayer_act_type": ["relu"]
}


cnnFrame = {
    "conv_kernel": ["1*18"],        # kernel height=1, width=18
    "conv_stride": ["1*1"],         # stride height=1, width=1
    "conv_num_filter": [4],         # 8 convolution filters

    "pool_act_type": ["relu"],      # ReLU after convolution
    "pool_type": ["max"],           # max pooling
    "pool_kernel": ["1*4"],         # pool over width=4
    "pool_stride": ["1*4"],         # stride of pooling = 4

    # Fully-connected layers: 32 units → 1 output neuron
    "fullayer_num_hidden": [32, 1],
    "fullayer_act_type": ["relu"],  # activation only for the first FC layer

    # Dropout: must be 1 more entry than number of FC layers
    # e.g. for FC = [32,1], you need drop_float = [drop_before_FC1, drop_before_FC2, drop_before_output]
    # "drop_float": [0.2, 0.1, 0.05]  # example values (tunable)
    # "drop_float": [0.4, 0.2, 0.1]  # example values (tunable)
    "drop_float": [0.1, 0.05, 0.025]  # example values (tunable)
}

# Marker image reshape (must match your genotype vector length)
# If your input width is e.g. 18 * 4 * some_blocks, set appropriately.
# Example:
# markerImage = (1, 784)  # JUST A PLACEHOLDER — update based on your data!
# markerImage = (1, 78)  # JUST A PLACEHOLDER — update based on your data!
# markerImage = (1, 2048)  # JUST A PLACEHOLDER — update based on your data!


# image reshaping (same format as R DeepGS)
# markerImage = (28, 28)