from matplotlib.pyplot import step
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from realpdebench.model.model import Model
from realpdebench.utils.metrics import mse_loss


class BranchNet(nn.Module):
    """Branch Net for processing 3D input functions, using 3D CNN structure"""
    def __init__(self, input_channels, p, dropout_rate=0.1):
        super().__init__()
        self.input_channels = input_channels
        self.dropout_rate = dropout_rate
        self.p = p
        
        # First convolution layer
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  
            #nn.MaxPool3d(2)
        )
       
        # Second convolution layer
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  
            #nn.MaxPool3d(2)
        )
        
       
        
        # Third convolution layer
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            #nn.MaxPool3d(2)  
        )
        
        
        # Fourth convolution layer
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 4, 4))
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256 * 1 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, p)
        )

    def forward(self, T0):
        # T0: [batch, T, H, W, C] - 3D input data, need to convert to [batch, C, T, H, W]
        batch_size, T, H, W, C = T0.shape
        # Rearrange dimensions to adapt to 3D convolution
        T0 = T0.permute(0, 4, 1, 2, 3)  # [batch, C, T, H, W]
        
        # Forward pass with residual connections
        x = self.conv1(T0)  
        x = self.conv2(x)   
        x = self.conv3(x)  
        x = self.conv4(x)   
        
        x = x.view(x.size(0), -1)
        return self.fc(x)  # [batch, p]

class TrunkNet(nn.Module):
    """Trunk Net for processing 3D coordinates, outputting p-dimensional basis functions"""
    def __init__(self, p, dropout_rate=0.1):
        super().__init__()
        # Optimized 3D coordinate structure, using deeper network
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, p),
            # nn.ReLU(),
            # nn.Dropout(dropout_rate), 
            # nn.Linear(256, p)
        )

    def forward(self, coords):
        # coords: [batch, num_points, 3] - 3D coordinates (t,x,y)
        return self.fc(coords)  # [batch, num_points, p]

class DeepONet(Model):
    """Complete DeepONet model"""
    def __init__(self, shape_in, shape_out, input_channels, output_channels, p, dropout_rate=0.1, device='cuda'):
        super().__init__()
        self.branch = BranchNet(input_channels, p, dropout_rate)
        self.trunk = TrunkNet(p, dropout_rate)
        # Simplified output network
        # self.output_net = nn.Linear(p, output_channels)
        self.output_net = nn.Sequential(
            nn.Linear(p, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128,output_channels)
        )
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.p = p
        self.device = device
     
    
    def forward(self, T0, delta_steps=None):
        # T0: [batch, T, H, W, C] - 3D input data
        batch_size, T_in, H, W, C = T0.shape
        T_out = self.shape_out[0]  # Output time steps
        
        # Generate 3D coordinate grid 
        #grid = self.get_grid((batch_size, T_out, H, W), T0.device)  # [batch, T_out, H, W, 3]
        grid = self.get_grid_at_time(batch_size, H, W, delta_steps, T0.device, T0.dtype)  # [batch, T_out, H, W, 3]
        coords = grid.reshape(grid.shape[0], -1, grid.shape[-1])  # [batch, T_out*H*W, 3]
        
        # Through branch network
        b = self.branch(T0)  # [batch, p]
        t = self.trunk(coords)  # [batch, num_points, p]
        output = self.output_net(b.unsqueeze(1) * t)  # [batch, num_points, output_channels]
        
        output = output.reshape(batch_size, T_out, H, W, -1)#[batch_size, T_out, H, W, output_channels]
        return output

    
    def train_loss(self, input, target):
        pred = self.forward(input)
        return mse_loss(pred, target)
        
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
    

    def _broadcast_delta(self, delta_steps, B, device, dtype):
        if delta_steps is None:
            return torch.ones(B, device=device, dtype=dtype)
        if isinstance(delta_steps, (int, float)):
            return torch.full((B,), float(delta_steps), device=device, dtype=dtype)
        delta_steps = torch.as_tensor(delta_steps, device=device, dtype=dtype)
        if delta_steps.dim() == 0:
            return delta_steps.repeat(B)
        if delta_steps.dim() == 1 and delta_steps.shape[0] == B:
            return delta_steps
        raise ValueError(
            f"delta_steps must be scalar or (B,), got {tuple(delta_steps.shape)}"
        )
    
    def get_grid_at_time(self, batch_size, H, W, delta_steps, device, dtype):
        """
        Build coordinates for one requested output time.

        Args:
            delta_steps: scalar or tensor of shape (batch_size,).

        Returns:
            grid: [batch_size, 1, H, W, 3], with coordinates (t, x, y).
        """
        steps   = self._broadcast_delta(delta_steps, batch_size, device, dtype)

        # Temporal coordinate: [B, 1, H, W, 1]
        grid_t = steps.view(batch_size, 1, 1, 1, 1).expand(batch_size, 1, H, W, 1)

        # Spatial coordinates: [B, 1, H, W, 1]
        grid_x = torch.linspace(0, 1, H, device=device, dtype=dtype)
        grid_x = grid_x.view(1, 1, H, 1, 1).expand(batch_size, 1, H, W, 1)

        grid_y = torch.linspace(0, 1, W, device=device, dtype=dtype)
        grid_y = grid_y.view(1, 1, 1, W, 1).expand(batch_size, 1, H, W, 1)

        return torch.cat((grid_t, grid_x, grid_y), dim=-1)