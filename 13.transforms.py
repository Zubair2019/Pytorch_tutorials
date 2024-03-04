from pathlib import Path
import torch
import matplotlib.pyplot as plt
plt.rcParams["savefig.bbox"] = 'tight'

from torchvision.transforms import v2
from torchvision.io import read_image

torch.manual_seed(1)

# If you're trying to run that on collab, you can download the assets and the
# helpers from https://github.com/pytorch/vision/tree/main/gallery/
from helpers import plot
img = read_image(str(Path('/Users/mac/Downloads') / 'astronaut.jpg'))
print(f"{type(img) = }, {img.dtype = }, {img.shape = }")

transform = v2.RandomCrop(size=(224,224))
out = transform(img)
plot([img, out])
print(f"{type(out) = }, {out.dtype = }, {out.shape = }")