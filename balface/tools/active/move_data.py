import os
import os.path as osp
import shutil


file = './datasets/FairFace/labels/cond-aug/fairface_14k_40k_craig_2k7.txt'

root = './datasets/FairFace/025_images'
dest_root = './datasets/FairFace/fairface_14k-40k_craig_2k7'

os.makedirs(dest_root, exist_ok=True)
cls_list = [
	'0-2',
    '3-9',
	'10-19',
	'20-29',
	'30-39',
	'40-49',
	'50-59',
	'60-69',
	'70+'
]

cls_mapping = {
	'White': 'White',
	'Middle Eastern': 'White',
	'Southeast Asian': 'East Asian',
	'East Asian': 'East Asian',
	'Black': 'Black',
	'Indian': 'Indian'
}

cls_list = ['White', 'Black', 'East Asian', 'Indian']

# cls_list = ['male', 'female']


with open(file, 'r') as f:
	for i, line in enumerate(f.readlines()):
		if i==0: continue
		line = line.strip().split(',')
		ori_im_name = line[0]
		im_name = line[0].split('/')[-1]
		cls = cls_mapping[line[1]]
		src_path = os.path.join(root, ori_im_name)
		dest_path = osp.join(osp.join(dest_root, cls), im_name)
		os.makedirs(osp.dirname(dest_path), exist_ok=True)
		shutil.copyfile(src_path, dest_path)
