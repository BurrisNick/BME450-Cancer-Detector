################
# covert the DCM images into PNG prior to data training
################

import pydicom
import numpy as np
from PIL import Image
from pathlib import Path

imnum = 1
for file in Path("cancer_data").rglob("*.dcm"):
    category = file.parent.name # determines what category of trials

    # 1. Load the DICOM
    ds = pydicom.dcmread(file)
    img = ds.pixel_array.astype(float)

    # 2. Normalize to 0-255 (Required for PNG)
    # Scaling ensures the highest value is white and lowest is black
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0

    # 3. Convert to 8-bit unsigned integers
    img_uint8 = img.astype(np.uint8)

    # 4. Save as PNG
    # If it's a standard scan, use 'L' mode for grayscale
    impath = 'cancer_data' + '/'+ f"cancer {imnum}.png"
    final_image = Image.fromarray(img_uint8, mode='L')
    final_image.save(Path(impath))
    imnum +=1