"""Model definitions, moved from Module 4 Practicals 1 (FCNN) and 3 (CNN)."""
import torch.nn as nn
import torch.nn.functional as F


class FCNN(nn.Module):
    """Fully connected network: flattens each 3x32x32 image into a 3072-long vector."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SimpleCNN(nn.Module):
    """First CNN from Practical 3: two conv + pool blocks, then two fully connected layers."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # 3 -> 16 channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)          # halves height and width
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16 -> 32 channels
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = x.view(-1, 32 * 8 * 8)            # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class EnhancedCNN(nn.Module):
    """Enhanced CNN from Practical 3: four conv layers with BatchNorm, plus Dropout."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv -> BatchNorm -> ReLU -> Pool, four times: 32 -> 16 -> 8 -> 4 -> 2
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(-1, 128 * 2 * 2)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


MODELS = {
    "FCNN": FCNN,
    "CNN": SimpleCNN,
    "EnhancedCNN": EnhancedCNN,
}


def get_model(model_name):
    """Return a new model by name: one of FCNN, CNN, EnhancedCNN (not case-sensitive)."""
    for name, model_class in MODELS.items():
        if name.lower() == model_name.lower():
            return model_class()
    raise ValueError(f"Unknown model '{model_name}'. Choose one of: {', '.join(MODELS)}")
