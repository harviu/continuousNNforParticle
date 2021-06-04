import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class GLO(nn.Module):
    def __init__(self, m_dim=1024):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(3, m_dim),
            nn.ReLU(True),
            nn.Linear(m_dim, m_dim),
            nn.ReLU(True),
            nn.Linear(m_dim, m_dim),
            nn.ReLU(True),
            nn.Linear(m_dim, m_dim),
            nn.ReLU(True),
            nn.Linear(m_dim, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        y = self.decoder(x)
        return y