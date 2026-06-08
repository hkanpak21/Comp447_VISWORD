import sys
from pathlib import Path
from PIL import Image

# Add src to python path
sys.path.append(str(Path('/scratch/bbakay22/VISWORD/src')))

from visword.data.cropper import NonOverlappingCropper, TextAwareCropper

# Create the output scratch directory if it doesn't exist
out_dir = Path('/scratch/bbakay22/VISWORD/scratch')
out_dir.mkdir(parents=True, exist_ok=True)

image_path = Path('/scratch/bbakay22/VISWORD/data/wiki_ss/blobs/00/0000000.png')
im = Image.open(image_path)

# 1. Old downsampled crop: crop_size=490, target_size=224
old_cropper = NonOverlappingCropper(crop_size=490, target_size=224)
old_crops = old_cropper(im)
if old_crops:
    old_crops[0].save(out_dir / 'old_crop_downsampled.png')
    print("Saved old crop downsampled:", old_crops[0].size)

# 2. New native legible crop: crop_size=224, target_size=224
new_cropper = TextAwareCropper(crop_size=224, target_size=224)
new_crops = new_cropper(im)
if new_crops:
    new_crops[0].save(out_dir / 'new_crop_native.png')
    print("Saved new crop native:", new_crops[0].size)

# 3. Pretraining full resolution crop: crop_size=490, target_size=490
pretrain_cropper = TextAwareCropper(crop_size=490, target_size=490)
pretrain_crops = pretrain_cropper(im)
if pretrain_crops:
    pretrain_crops[0].save(out_dir / 'pretrain_crop_490.png')
    print("Saved pretrain crop 490:", pretrain_crops[0].size)
