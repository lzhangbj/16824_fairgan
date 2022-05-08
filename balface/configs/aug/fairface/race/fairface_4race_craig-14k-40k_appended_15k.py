gpus=(0,1,2,3,5,6,7)

data = dict(
    train=dict(
        dataset="cls_fairface_4races_025",
        root="./datasets/FairFace/025_images",
        n_classes=[4,],
        face_label_txt="./datasets/FairFace/labels/cond-aug/fairface_14k_40k_craig_15k_appended.txt",
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
		# face_label_txt="./datasets/FairFace/labels/all/fairface__all_label.txt",
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
    name='cls_recognizer',
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
    use_weight=False
)


trainer = dict(
    name='cls',
    max_epochs=50,
    save_freq=5,
    print_freq=10,
    val_freq=5,
    loss_weight='none',
    sample_ratio=[13736, 5272, 5446, 4542.]
)

optimizer = dict(
    name='adam',
    lr=0.0005,
    weight_decay=0.0001
)

lr_scheduler = dict(
    name='multisteplr',
    milestones=[20, 45],
    # warmup_epoch=2,
    # warmup_lr=0.00001
)

find_unused_parameters = False

resume_from = None # 'work_dirs/fairface_4race_imb_14k_aug_40k_eval-label/epoch_50.pth'
load_backbone = None  # 'work_dirs/ssp_fakemake-100000_resnet34_adam/epoch_40.pth'
load_from = None # 'work_dirs/race_cls_fairface_4race025-14000-white-10760-1080-balanced_resnet34_adam/epoch_50.pth'