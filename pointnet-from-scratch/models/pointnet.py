import torch
import torch.nn as nn

class TNet(nn.Module):
    """
    T-Net (Transformation Network) module from the PointNet paper.
    
    The T-Net module learns an affine transformation matrix (k x k) from the input
    point set or intermediate features. This transformation aligns the input data
    (e.g., coordinates or features) to a canonical space before further processing,
    enforcing invariance to geometric transformations such as rotation, translation, and scaling.
    """
    def __init__(self, k: int = 3):
        super(TNet, self).__init__()
        self.k = k
        
        # Shared MLP: k -> 64 -> 128 -> 1024
        # We use nn.Conv1d for the shared MLP since the input is of shape (B, k, N)
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # Fully connected layers: 1024 -> 512 -> 256 -> k*k
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, k * k)
        
        self.relu = nn.ReLU()
        
        # Initialize the weights of the final FC layer to 0 and its bias to the
        # flattened identity matrix. This is the "PointNet trick" so the network
        # initially predicts an identity transform (i.e. leaves inputs unchanged)
        # and then learns residuals/refinements on top of it.
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(k).view(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape x: (batch_size, k, num_points)
        batch_size = x.size(0)
        
        # Shared MLP layers
        # x shape changes: (B, k, N) -> (B, 64, N)
        x = self.relu(self.bn1(self.conv1(x)))
        # x shape changes: (B, 64, N) -> (B, 128, N)
        x = self.relu(self.bn2(self.conv2(x)))
        # x shape changes: (B, 128, N) -> (B, 1024, N)
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Global max pooling over the point dimension
        # x shape changes: (B, 1024, N) -> (B, 1024, 1) -> (B, 1024)
        x = torch.max(x, dim=2)[0]
        
        # Fully connected layers
        # x shape changes: (B, 1024) -> (B, 512)
        x = self.relu(self.bn4(self.fc1(x)))
        # x shape changes: (B, 512) -> (B, 256)
        x = self.relu(self.bn5(self.fc2(x)))
        # x shape changes: (B, 256) -> (B, k*k)
        x = self.fc3(x)
        
        # Reshape to batch of k x k matrices
        # x shape changes: (B, k*k) -> (B, k, k)
        transform = x.view(batch_size, self.k, self.k)
        
        return transform


class PointNetClassifier(nn.Module):
    """
    PointNet classification network architecture from the paper:
    "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" (Qi et al., 2017).
    
    The classification pipeline is as follows:
    1. Input transform: TNet(k=3) applied to raw input points, then batch matrix multiply to align input.
    2. Shared MLP: 3 -> 64 -> 64 (Conv1d + BatchNorm1d + ReLU each).
    3. Feature transform: TNet(k=64) applied to the 64-dim features (only if feature_transform=True), then batch matrix multiply.
    4. Shared MLP: 64 -> 64 -> 128 -> 1024 (Conv1d + BatchNorm1d + ReLU each).
    5. Global max pooling over points to extract a 1024-dim global feature.
    6. FC classifier head: 1024 -> 512 -> 256 -> num_classes, with BatchNorm1d, ReLU, and Dropout(0.3) on the first two FC layers.
    """
    def __init__(self, num_classes: int = 40, feature_transform: bool = True):
        super(PointNetClassifier, self).__init__()
        self.feature_transform = feature_transform
        
        # 1. Input transform network
        self.input_transform_net = TNet(k=3)
        
        # 2. Shared MLP 1: 3 -> 64 -> 64
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        
        # 3. Feature transform network (optional)
        if self.feature_transform:
            self.feature_transform_net = TNet(k=64)
            
        # 4. Shared MLP 2: 64 -> 64 -> 128 -> 1024
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn5 = nn.BatchNorm1d(1024)
        
        # 6. Fully connected classifier head
        self.fc1 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # Input shape: (B, N, 3)
        batch_size = x.size(0)
        num_points = x.size(1)
        
        # Transpose input to match Conv1d expectation: (B, N, 3) -> (B, 3, N)
        x = x.transpose(2, 1)
        # x: (B, 3, N)
        
        # 1. Input transform
        trans = self.input_transform_net(x)
        # trans: (B, 3, 3)
        x = torch.bmm(trans, x)
        # x: (B, 3, N)
        
        # 2. Shared MLP: 3 -> 64 -> 64
        x = self.relu(self.bn1(self.conv1(x)))
        # x: (B, 64, N)
        x = self.relu(self.bn2(self.conv2(x)))
        # x: (B, 64, N)
        
        # 3. Feature transform (optional)
        if self.feature_transform:
            trans_feat = self.feature_transform_net(x)
            # trans_feat: (B, 64, 64)
            x = torch.bmm(trans_feat, x)
            # x: (B, 64, N)
        else:
            trans_feat = None
            
        # 4. Shared MLP: 64 -> 64 -> 128 -> 1024
        x = self.relu(self.bn3(self.conv3(x)))
        # x: (B, 64, N)
        x = self.relu(self.bn4(self.conv4(x)))
        # x: (B, 128, N)
        x = self.relu(self.bn5(self.conv5(x)))
        # x: (B, 1024, N)
        
        # 5. Global Max Pooling
        x = torch.max(x, dim=2)[0]
        # x: (B, 1024)
        
        # 6. FC Classifier Head
        x = self.relu(self.bn6(self.fc1(x)))
        # x: (B, 512)
        x = self.dropout(x)
        # x: (B, 512)
        x = self.relu(self.bn7(self.fc2(x)))
        # x: (B, 256)
        x = self.dropout(x)
        # x: (B, 256)
        
        logits = self.fc3(x)
        # logits: (B, num_classes)
        
        return logits, trans, trans_feat

