import torch
from visword.models.clip_backbone import CLIPImageBackbone
from PIL import Image
import numpy as np

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPImageBackbone(num_trainable_blocks=0).to(device).eval()
    
    x1 = torch.randn(1, 3, 224, 224).to(device)
    x2 = torch.randn(1, 3, 224, 224).to(device)
    
    v = model.visual
    
    with torch.no_grad():
        y1 = model._renorm(x1)
        y2 = model._renorm(x2)
        
        y1 = v.conv1(y1)
        y2 = v.conv1(y2)
        
        B, C, H, W = y1.shape
        y1 = y1.reshape(B, C, H * W).permute(0, 2, 1)
        y2 = y2.reshape(B, C, H * W).permute(0, 2, 1)
        
        cls = v.class_embedding.to(y1.dtype) + torch.zeros(B, 1, C, dtype=y1.dtype, device=y1.device)
        y1 = torch.cat([cls, y1], dim=1)
        y2 = torch.cat([cls, y2], dim=1)
        
        y1 = y1 + v.positional_embedding.to(y1.dtype)
        y2 = y2 + v.positional_embedding.to(y2.dtype)
        
        y1 = v.ln_pre(y1)
        y2 = v.ln_pre(y2)
        
        y1 = y1.permute(1, 0, 2)
        y2 = y2.permute(1, 0, 2)
        
        print("Starting blocks...")
        for i, blk in enumerate(v.transformer.resblocks):
            # THIS IS WHAT CLIPImageBackbone.forward DOES
            with torch.no_grad():
                y1 = blk(y1)
                y2 = blk(y2)
            print(f"Block {i} diff: {(y1-y2).abs().max().item()}")

if __name__ == "__main__":
    main()
