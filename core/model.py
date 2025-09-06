# models/cnn_model.py

import torch.nn as nn


class CNNModel(nn.Module):
    """轻量级CNN模型 - 防止过拟合版本"""

    def __init__(self, num_classes, input_channels=1, base_filters=16, num_conv_blocks=3, dropout_rate=0.3):
        super(CNNModel, self).__init__()

        layers = []
        in_channels = input_channels
        out_channels = base_filters

        # 添加卷积块（减少层数和参数）
        for i in range(num_conv_blocks):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate * 0.5),  # 在卷积层添加dropout
                nn.MaxPool2d(kernel_size=2)
            ])
            in_channels = out_channels
            out_channels = min(out_channels * 2, 128)  # 限制最大通道数

        self.features = nn.Sequential(*layers)

        # 更简单的分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_channels, 64),  # 减少隐藏层大小
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x