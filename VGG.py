import torch
import torch.nn as nn 

# Custom exception for CUDA requirement
class CUDANotAvailableError(Exception):
    """Exception raised when CUDA is required but not available."""
    pass

class VGG16(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        super(VGG16, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features = self.layers()
        # Adaptive avg pooling for flexible input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Optimized classifier for CIFAR-10 (smaller FC layers)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),# 1 fc layer with 512 units (reduced from 4096)
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        # Enforce CUDA usage
        if not torch.cuda.is_available():
            raise CUDANotAvailableError(
                "CUDA is not available! CUDA is required for training this model. "
                "Please ensure you have a CUDA-enabled GPU and PyTorch with CUDA support installed."
            )
        self.device = torch.device("cuda")
        self.to(self.device)
        print(f"CUDA enabled: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    def layers(self):
        layers = []
        in_channels = 3
        # 'M' stands for max pooling layer
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        for x in cfg:
            if x == 'M':
                layers += [self.maxpool]
            else:
                layers += [
                    nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features=x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        return nn.Sequential(*layers)
    
    def get_device(self):
        return self.device
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)  # Adaptive avg pooling: (batch, 512, 1, 1)
        x = torch.flatten(x, 1)  # Flatten to (batch, 512)
        x = self.classifier(x)
        return x
