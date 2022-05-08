lines = []

with open('furthest_sample_14k.txt', 'r') as f:
	for i,line in enumerate(f.readlines()):
		if i==0: continue
		line = line.strip().split(',')
		image_name = line[0]
		race = line[1]
		if "fairface_race_imb_14k" in image_name:
			image_name = '/'.join(image_name.split('/')[-2:])
		else:
			image_name = '/'.join(image_name.split('/')[-3:])
		lines.append(f"{image_name},{race}\n")

with open('furthest_sample_14k.txt', 'w') as f:
	f.writelines("image_name,race\n")
	for line in lines:
		f.writelines(line)


		