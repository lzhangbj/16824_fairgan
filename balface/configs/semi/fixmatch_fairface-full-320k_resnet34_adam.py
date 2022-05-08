gpus=(0,1,2,3,5,6,7)

data = dict(
    labeled_train=dict(
        dataset="cls_fairface_4races_025",
        root="./datasets/FairFace/025_images",
        n_classes=[4,],
        size=224,
        face_label_txt="./datasets/FairFace/labels/race/fairface_train_4race-7000-white-5380-540-balanced_label.txt",
        samples_per_gpu=128,
        workers_per_gpu=4,
        mode='train'
    ),
    unlabeled_train=dict(
        dataset="fixmatch_025",
        root="./datasets/FairFace/025_images",
        size=224,
	    face_label_txt="./datasets/FairFace/labels/cond-aug/stylegan2_generated_80k.txt",
	    samples_per_gpu=384, # 500
        workers_per_gpu=8,
        mode='train'
    ),
    val=dict(
        dataset="cls_fairface_4races_025",
        root="./datasets/FairFace/025_images",
        size=224,
        n_classes=[4,],
        face_label_txt="./datasets/FairFace/labels/all/fairface_val_all_label.txt",
        samples_per_gpu=256,
        workers_per_gpu=8,
        mode='val'
    )
)

model = dict(
    name="fixmatch_recognizer",
    backbone=dict(
        name="torch_resnet34",
    ),
    head=dict(
        name="multiclassifier_head",
        n_classes=[4,],
        input_dim=512,
        hidden_dims=[],
		loss = 'ce',
        norm_weights = False
    ),
    use_weight=False,
    ema_momentum=0.95,
    loss='ce'
)

trainer = dict(
    name='fixmatch',
    max_epochs=60,
    save_freq=5,
    print_freq=10,
    val_freq=5,
    cls_weight=0.1,
    ssl_weight=2.0,
    resample=None,
	adapt_pseudo_weights=False,
	use_weight=False
)

optimizer = dict(
    name='adam',
    lr=0.0005,
    weight_decay=0.0001
)

lr_scheduler = dict(
    name='multisteplr',
    milestones=[30,]
)


find_unused_parameters = False

resume_from = 'work_dirs/fixmatch_fairface-full-320k_resnet34_adam/epoch_50.pth'
load_backbone = None
load_from = None
