import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class AD_SDF(nn.Module):
    def __init__(self, z_dim=16, m_dim=256):
        super(AD_SDF, self).__init__()
        self.decoder_stage1 = nn.Sequential(
            nn.Linear(z_dim+3, m_dim),
            nn.ReLU(True),
            # nn.Linear(m_dim, m_dim),
            # nn.ReLU(True),
            # nn.Linear(m_dim, m_dim),
            # nn.ReLU(True),
            nn.Linear(m_dim, m_dim-3-z_dim),
            nn.ReLU(True),)
        
        self.decoder_stage2 = nn.Sequential(
            nn.Linear(m_dim, m_dim),
            nn.ReLU(True),
            # nn.Linear(m_dim, m_dim),
            # nn.ReLU(True),
            # nn.Linear(m_dim, m_dim),
            # nn.ReLU(True),
            nn.Linear(m_dim, 1))
    
    
    def forward(self, code):
        decoder_stage1_out = self.decoder_stage1(code)
        code = torch.cat((decoder_stage1_out, code), dim=1)
        decoder_stage2_out = self.decoder_stage2(code)
        return decoder_stage2_out
    