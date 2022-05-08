gpus=(0,1,2,3,5,6,7)

data = dict(
    train=dict(
        dataset="cls_fairface_4races_025",
        root="./datasets/FairFace/025_images",
        n_classes=[4,],
        face_label_txt="./datasets/FairFace/labels/cond-aug/furthest_cls_sample_v2_14k.txt",
        size=224,
        samples_per_gpu=256,
        workers_per_gpu=8,
        mode='train'
    ),
	val=dict(
		dataset="cls_fairface_4races_025",
		root="./datasets/FairFace/025_images",
		size=224,
		n_classes=[4, ],
		face_label_txt="./datasets/FairFace/labels/race/fairface_train_val_4race-1000_label.txt",
		samples_per_gpu=256,
		workers_per_gpu=8,
		mode='val'
	),
	test=dict(
		dataset="cls_fairface_4races_025",
		root="./datasets/FairFace/025_images",
		size=224,
		n_classes=[4, ],
		face_label_txt="./datasets/FairFace/labels/all/fairface_val_all_label.txt",
		samples_per_gpu=256,
		workers_per_gpu=8,
		mode='val'
	)
)

model = dict(
    name='cls_featmix_recognizer',
    backbone=dict(
        name="torch_resnet34"
    ),
    head=dict(
        name="multiclassifier_head",
        n_classes=[4,],
        input_dim=512,
        hidden_dims=[],
        loss='ce',
        norm_weights=False
    ),
    use_weight=False,
	aug_method='intra_rand',
	aug_ratio=[1., 1, 1, 1],
	aug_num_per_gpu=256
)


trainer = dict(
    name='cls',
    max_epochs=60,
    save_freq=5,
    print_freq=10,
    val_freq=5,
    loss_weight='none',
    sample_ratio=[1, 1, 1, 1.]
)

optimizer = dict(
    name='adam',
    lr=0.0005,
    weight_decay=0.0001
)

lr_scheduler = dict(
    name='multisteplr',
    milestones=[20, 50],
    # warmup_epoch=2,
    # warmup_lr=0.00001
)

find_unused_parameters = False

resume_from = None #'work_dirs/race_cls_fairface_hyperstyle-4race025_v7-top-10-100_10-10_white-0-100-balanced_resnet34_adam/epoch_48.pth'
load_backbone = None  # 'work_dirs/ssp_fakemake-100000_resnet34_adam/epoch_40.pth'
load_from = None # 'work_dirs/race_cls_fairface_imb_14k_reweight_resnet34_adam/best_model.pth'