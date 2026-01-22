import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SelfAttention1D, SelfAttention2D, LogitBiasedSelfAttention1D

attention_classes = {
    "self": lambda channels, num_heads, dropout: SelfAttention1D(channels, num_heads, dropout),
    "sqi": lambda channels, num_heads, dropout: LogitBiasedSelfAttention1D(channels, num_heads, dropout)
}

class BasicBlock2D(nn.Module):
    expansion = 1  # No expansion in BasicBlock

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_p=0.2):
        super(BasicBlock2D, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_p)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # Downsampling layer if needed

    def forward(self, x):
        identity = x  # Save input for residual connection
        if self.downsample is not None:
            identity = self.downsample(x)  # Adjust dimensions if needed

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Residual connection
        out = self.relu(out)

        return out

class BasicBlock1D(nn.Module):
    expansion = 1  # No expansion in BasicBlock

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_p=0.2):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_p)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # Downsampling layer if needed

    def forward(self, x):
        identity = x  # Save input for residual connection
        if self.downsample is not None:
            identity = self.downsample(x)  # Adjust dimensions if needed

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Residual connection
        out = self.relu(out)

        return out


class ResNet2D(nn.Module):
    def __init__(self, block, num_layers, signal_channels, layer_norm, feat_dim, dropout_p,
                 attention_heads=8, use_attention=True):
        super(ResNet2D, self).__init__()

        assert num_layers in [10, 18, 34], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                           f'to be 10, 18, or 34 '

        if num_layers == 10:
            layers = [1, 1, 1, 1]
        elif num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34:
            layers = [3, 4, 6, 3]

        self.in_channels = 64  # Initial channels before block stacking
        self.layer_norm = layer_norm
        self.feat_dim = feat_dim

        # Initial Convolutional Layer (Conv + BN + ReLU + MaxPool)
        self.conv1 = nn.Conv2d(signal_channels, 64, kernel_size=6, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Blocks
        self.layer1 = self._make_layer(block, 64, layers[0], dropout_p, stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], dropout_p, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], dropout_p, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], dropout_p, stride=2)

        self.use_attention = use_attention
        if self.use_attention:
            self.attention = SelfAttention2D(512 * block.expansion, num_heads=attention_heads, dropout=dropout_p)

        # Global Average Pooling & Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, self.feat_dim)

        if self.layer_norm:
            self.feat_norm_layer = nn.LayerNorm(self.feat_dim)
        # Initialize Weights
        # self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, dropout_p, stride):
        """Create a ResNet layer with multiple residual blocks."""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout_p))  # First block may downsample
        self.in_channels = out_channels * block.expansion  # Update channels

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1, downsample=None, dropout_p=dropout_p))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights with Kaiming He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Define forward pass of ResNet."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.use_attention:
            # Save feature map before attention
            feat_map = x  # (B, C, H, W)
            # Apply attention
            attended_feat_map = self.attention(x)
            # Extract embeddings: mean over spatial dimensions (H, W)
            self.z_backbone = torch.mean(feat_map.view(feat_map.size(0), feat_map.size(1), -1), dim=-1)  # (B, C)
            self.z_att = torch.mean(attended_feat_map.view(attended_feat_map.size(0), attended_feat_map.size(1), -1), dim=-1)  # (B, C)
            x = attended_feat_map
        else:
            # If no attention, still save backbone embedding
            feat_map = x
            self.z_backbone = torch.mean(feat_map.view(feat_map.size(0), feat_map.size(1), -1), dim=-1)  # (B, C)
            self.z_att = None

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.layer_norm:
            x = self.feat_norm_layer(x)

        return x

class ResNet1D(nn.Module):
    def __init__(self, block, num_layers, signal_channels, layer_norm, feat_dim, dropout_p,
                 attention_heads=8, use_attention=True, attention_type="self"):
        super(ResNet1D, self).__init__()

        assert num_layers in [10, 18, 34], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                           f'to be 18, or 34 '

        if num_layers == 10:
            layers = [1, 1, 1, 1]
        elif num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34:
            layers = [3, 4, 6, 3]

        self.in_channels = 64  # Initial channels before block stacking
        self.layer_norm = layer_norm
        self.feat_dim = feat_dim
        self.attention_type = attention_type

        # Initial Convolutional Layer (Conv + BN + ReLU + MaxPool)
        self.conv1 = nn.Conv1d(signal_channels, 64, kernel_size=6, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet Blocks
        self.layer1 = self._make_layer(block, 64, layers[0], dropout_p, stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], dropout_p, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], dropout_p, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], dropout_p, stride=2)

        self.use_attention = use_attention
        if self.use_attention:
            self.attention = attention_classes[self.attention_type](512 * block.expansion, num_heads=attention_heads, dropout=dropout_p)

        # Global Average Pooling & Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, self.feat_dim)

        if self.layer_norm:
                self.feat_norm_layer = nn.LayerNorm(self.feat_dim)
        # Initialize Weights
        # self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, dropout_p, stride):
        """Create a ResNet layer with multiple residual blocks."""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout_p))  # First block may downsample
        self.in_channels = out_channels * block.expansion  # Update channels

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1, downsample=None, dropout_p=dropout_p))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights with Kaiming He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, sqi_sample=None):
        """
        Define forward pass of ResNet.
        
        Args:
            x: Input tensor of shape (B, C, T)
            sqi_sample: SQI sample tensor of shape (B, T_orig) - temporal SQI matching input signal length
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.use_attention:
            # Save feature map before attention
            feat_map = x  # (B, C, T)
            
            if self.attention_type == "sqi" and sqi_sample is not None:
                # Process sqi_sample to align with features after layer 4
                # Get actual feature map temporal dimension (may not be exactly T_orig/32 due to padding)
                B, C, T_feat = x.shape
                
                # Convert sqi_sample to tensor if needed
                if not isinstance(sqi_sample, torch.Tensor):
                    sqi_sample = torch.from_numpy(sqi_sample).float()
                
                # Move to same device as x
                sqi_sample = sqi_sample.to(x.device)
                
                # Process sqi_sample for each sample in the batch
                # Use adaptive average pooling to match exact T_feat length
                sqi_feat_list = []
                for b in range(B):
                    sqi_orig = sqi_sample[b]  # (T_orig,)
                    
                    # Use adaptive average pooling to exactly match T_feat
                    # Reshape to (1, 1, T_orig) for pooling
                    sqi_orig_2d = sqi_orig.unsqueeze(0).unsqueeze(0)  # (1, 1, T_orig)
                    # Adaptive average pool to exact target length
                    sqi_feat = F.adaptive_avg_pool1d(sqi_orig_2d, output_size=T_feat)
                    sqi_feat = sqi_feat.squeeze(0).squeeze(0)  # (T_feat,)
                    
                    sqi_feat_list.append(sqi_feat)
                
                # Stack into batch tensor: (B, T_feat)
                sqi_feat_batch = torch.stack(sqi_feat_list)  # (B, T_feat)
                
                # Verify shape matches
                assert sqi_feat_batch.shape == (B, T_feat), \
                    f"SQI feature shape {sqi_feat_batch.shape} doesn't match expected {(B, T_feat)}"
                
                # Apply attention with aligned SQI
                attended_feat_map = self.attention(x, sqi_feat_batch)
            elif self.attention_type == "sqi":
                # SQI not provided, skip attention
                attended_feat_map = feat_map
            else:
                attended_feat_map = self.attention(x)
            
            # Extract embeddings: mean over temporal dimension (T)
            self.z_backbone = torch.mean(feat_map, dim=-1)  # (B, C)
            self.z_att = torch.mean(attended_feat_map, dim=-1)  # (B, C)
            x = attended_feat_map
        else:
            # If no attention, still save backbone embedding
            feat_map = x
            self.z_backbone = torch.mean(feat_map, dim=-1)  # (B, C)
            self.z_att = None

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.layer_norm:
            x = self.feat_norm_layer(x)

        return x

def resnet18_2D(signal_channels, stride, layer_norm, feat_dim, dropout_p, use_attention=True, attention_heads=8):
    return ResNet2D(
        BasicBlock2D,
        18,
        signal_channels=signal_channels,
        layer_norm=layer_norm,
        feat_dim=feat_dim,
        dropout_p=dropout_p,
        use_attention=use_attention,
        attention_heads=attention_heads,
    )

def resnet10_2D(signal_channels, stride, layer_norm, feat_dim, dropout_p, use_attention=True, attention_heads=8):
    return ResNet2D(
        BasicBlock2D,
        10,
        signal_channels=signal_channels,
        layer_norm=layer_norm,
        feat_dim=feat_dim,
        dropout_p=dropout_p,
        use_attention=use_attention,
        attention_heads=attention_heads,
    )

def resnet34_2D(signal_channels, stride, layer_norm, feat_dim, dropout_p, use_attention=True, attention_heads=8):
    return ResNet2D(
        BasicBlock2D,
        34,
        signal_channels=signal_channels,
        layer_norm=layer_norm,
        feat_dim=feat_dim,
        dropout_p=dropout_p,
        use_attention=use_attention,
        attention_heads=attention_heads,
    )

def resnet18_1D(signal_channels, stride, layer_norm, feat_dim, dropout_p, use_attention=True, attention_heads=8, attention_type="self"):
    return ResNet1D(
        BasicBlock1D,
        18,
        signal_channels=signal_channels,
        layer_norm=layer_norm,
        feat_dim=feat_dim,
        dropout_p=dropout_p,
        use_attention=use_attention,
        attention_heads=attention_heads,
        attention_type=attention_type,
    )

def resnet34_1D(signal_channels, stride, layer_norm, feat_dim, dropout_p, use_attention=True, attention_heads=8, attention_type="self"):
    return ResNet1D(
        BasicBlock1D,
        34,
        signal_channels=signal_channels,
        layer_norm=layer_norm,
        feat_dim=feat_dim,
        dropout_p=dropout_p,
        use_attention=use_attention,
        attention_heads=attention_heads,
        attention_type=attention_type,
    )