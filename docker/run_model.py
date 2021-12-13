import json
import os
import re
from glob import glob
from pathlib import Path
from tqdm import tqdm

from PIL import Image
import numpy as np
from torchgeo.trainers import RegressionTask
import torch

jpeg_pattern = re.compile(r".*\.jpe?g$")
EXPECTED_SIZE = 366

def get_input_tensor(fp: str) -> torch.Tensor:
    with Image.open(fp) as img:
        if img.height != EXPECTED_SIZE or img.width != EXPECTED_SIZE:
            img = img.resize(size=(EXPECTED_SIZE, EXPECTED_SIZE), resample=Image.BILINEAR)
        array = np.array(img)
        if len(array.shape) == 3:
            array = array[:, :, 0]
        tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
        tensor = tensor / 255.
        tensor = tensor.unsqueeze(0).repeat(1, 3, 1, 1)
        return tensor

# This is set in the Dockerfile
checkpoint_location = os.environ["CHECKPOINT_LOCATION"]
input_volume=Path(os.environ["INPUT_DATA_VOLUME"])
output_volume=Path(os.environ["OUTPUT_DATA_VOLUME"])

# Determine the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the model from the checkpoint file
model = RegressionTask(
    model="resnet18",
    pretrained=False,
)
model = model.load_from_checkpoint(checkpoint_location)
model.freeze()
model = model.eval()
model = model.to(device)

# Get the list of images
all_files = glob(str(Path(input_volume) / "**" / "*"))
image_files = [f for f in all_files if jpeg_pattern.match(f) is not None]
for fp in tqdm(image_files):
    # Load the image, convert to tensor, and normalize
    tensor = get_input_tensor(fp)
    
    # Generate prediction
    with torch.no_grad():
        output = model(tensor).cpu().numpy().item(0)
        output = round(output)

    # Get the output path
    relative_path = Path(fp).relative_to(input_volume)
    output_path = (output_volume / relative_path).with_suffix(".json")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with output_path.open("w") as dst:
        json.dump({"wind_speed": output}, dst)