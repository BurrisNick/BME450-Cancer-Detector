################
# covert the DCM images into PNG prior to data training
################

import pydicom
import numpy as np
from PIL import Image
from pathlib import Path

imnum = 1
<<<<<<< Updated upstream
for file in Path("cancer_data" + '/' + 'idk what data this is').rglob("*.dcm"):
    category = file.parent.name # determines what category of trials
    print(file)
    # 1. Load the DICOM
    ds = pydicom.dcmread(file)

    print(ds.pixel_array)

    if ds.pixel_array.ndim < 2 or ds.pixel_array.shape[-1] < 100:
        print(f"Skipping {file.name}: Not a viewable image (Shape: {ds.pixel_array.shape})")
        continue

    img = ds.pixel_array.astype(float)

    img = np.squeeze(img)

=======
for file in Path("cancer_data").rglob("*.dcm"):
    category = file.parent.name # determines what category of trials

    # 1. Load the DICOM
    ds = pydicom.dcmread(file)
    img = ds.pixel_array.astype(float)

>>>>>>> Stashed changes
    # 2. Normalize to 0-255 (Required for PNG)
    # Scaling ensures the highest value is white and lowest is black
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0

    # 3. Convert to 8-bit unsigned integers
    img_uint8 = img.astype(np.uint8)

    # 4. Save as PNG
    # If it's a standard scan, use 'L' mode for grayscale
<<<<<<< Updated upstream
    impath = 'cancer_data' + '/' 'pngs of cancer' + '/' + f"cancer {imnum}.png"
=======
    impath = 'cancer_data' + '/'+ f"cancer {imnum}.png"
>>>>>>> Stashed changes
    final_image = Image.fromarray(img_uint8, mode='L')
    final_image.save(Path(impath))
    imnum +=1