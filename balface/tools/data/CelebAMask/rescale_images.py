import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm


hires_images = glob("1024x1024/*.jpg")

for image in tqdm(hires_images):
    im = Image.open(image)
    im = im.resize((512, 512))

    image = image.replace('1024x1024', 'images')
    im.save(image)