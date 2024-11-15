import torch
import torch.nn as nn
from einops.einops import rearrange

from .position_encoding import PositionEncodingSine
from .transformer import LocalFeatureTransformer

class LoFTR(nn.Module):
    def __init__(self):
        super().__init__()

        self.pos_encoding = PositionEncodingSine(
            d_model=128,
            temp_bug_fix=False)
        self.loftr_coarse = LocalFeatureTransformer()

    def forward(self, feat_c0, feat_c1):
 
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n h w c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n h w c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        return feat_c0, feat_c1
