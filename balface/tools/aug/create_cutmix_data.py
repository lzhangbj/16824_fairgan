import os
import os.path as osp
import random
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
from balface.utils import GaussianBlur

cls_mapping = {
	'White': 'White',
	'Middle Eastern': 'White',
	'Black': 'Black',
	'East Asian': 'East Asian',
	'Southeast Asian': 'East Asian',
	'Indian': 'Indian'
}

cls_list = [
	'White', 'Black', 'East Asian', 'Indian'
]

def read_label(file):
	cls_image_list = {
		'White': [],
		'Black': [],
		'East Asian': [],
		'Indian': []
	}
	with open(file, 'r') as f:
		for i, line in enumerate(f.readlines()):
			if i==0: continue
			line = line.strip().split(',')
			image_name = line[0]
			race = line[1]
			if race == 'Latino_Hispanic': continue
			race = cls_mapping[race]
			cls_image_list[race].append(image_name)
	return cls_image_list


def cutmix_images(pil_img1, pil_img2, coeff, size=256):
	'''
		crop coeff area part of image 1 to paste onto image2
	'''
	rw = int(size * np.sqrt(coeff))
	rh = rw
	
	rx = np.random.randint(size + 1 - rw)
	ry = np.random.randint(size + 1 - rh)
	
	cropped = pil_img1.crop((rx, ry, rx + rw, ry + rh))
	pil_img2.paste(cropped, (rx, ry, rx + rw, ry + rh))
	
	return pil_img2


if __name__ == '__main__':
	from tqdm import tqdm
	image_root1 = './datasets/FairFace/025_images'
	label_file1 = './datasets/FairFace/labels/race/fairface_train_4race-14000-white-10760-1080-balanced_label.txt'
	image_root2 = './datasets/FairFace/025_images'
	label_file2 = './datasets/FairFace/labels/cond-aug/4race_imb_14k_aug_40k.txt'
	save_image_dir = './datasets/FairFace/025_images/race_cutmix_intra_imb-14k_x1'
	save_label_file = './datasets/FairFace/labels/mix-aug/race_cutmix_intra_imb-14k_x1.txt'
	
	cls_image_list1 = read_label(label_file1)
	cls_image_list2 = read_label(label_file2)
	
	save_image_prefix = save_image_dir.split('/')[-1]
	
	np.random.seed(123)
	
	image_num_per_inst = 1
	with open(save_label_file, 'w') as ff:
		ff.writelines("image_name,race")
		for cls in cls_list:
			print(f"############### processing class {cls} ##################")
			save_cls_dir = osp.join(save_image_dir, cls)
			os.makedirs(save_cls_dir, exist_ok=True)
			cand_image_list = cls_image_list1[cls]
			for image_name in tqdm(cls_image_list1[cls]):
				image_path = osp.join(image_root1, image_name)
				with open(image_path, 'rb') as f:
					master_image = Image.open(f).convert('RGB').resize((256, 256))
				master_image_name = image_name.split('/')[-1].replace(".jpg", '')
				
				slave_image_indices = np.random.choice(len(cand_image_list), size=image_num_per_inst, replace=False)
				for slave_index in slave_image_indices:
					slave_image_name = cand_image_list[slave_index]
					slave_image_path = osp.join(image_root2, slave_image_name)
					with open(slave_image_path, 'rb') as f:
						slave_image = Image.open(f).convert('RGB').resize((256, 256))
					slave_image_name = slave_image_name.split('/')[-1].replace(".jpg", '')
					
					mix_image_name = f'{cls}_{master_image_name}_{slave_image_name}.jpg'
					
					coeff = np.random.uniform(0.7, 1.0)
					cutmix_image = cutmix_images(master_image, slave_image, coeff, size=256)
					cutmix_image.save(osp.join(save_cls_dir, mix_image_name))
					ff.writelines(f"\n{save_image_prefix}/{cls}/{mix_image_name},{cls}")
			
				
				
	
	

