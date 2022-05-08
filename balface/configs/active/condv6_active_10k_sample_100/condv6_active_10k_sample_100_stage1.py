gpus=(0,1,2,3,5,6,7)

data = dict(
    train=dict(
        dataset="cls_fairface_4races_025_aug",
        root="./datasets/FairFace/025_images",
        n_classes=[4,],
        face_label_txt="./datasets/FairFace/labels/race/fairface_train_4race-7000-white-5380-540-balanced_label.txt",
		aug_face_label_txt="./datasets/FairFace/labels/cond-aug/condv6_active_10k_sample_100_stage1.txt",
        size=224,
	    weights_list=[0.1, ],
        samples_per_gpu=256,
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
    max_epochs=10,
    save_freq=1,
    print_freq=10,
    val_freq=1,
    loss_weight='none',
    sample_ratio=[10, 1, 1, 1.]
)

optimizer = dict(
    name='adam',
    lr=0.00005,
    weight_decay=0.0001
)

lr_scheduler = dict(
    name='multisteplr',
    milestones=[],
    # warmup_epoch=2,
    # warmup_lr=0.00001
)

find_unused_parameters = False

resume_from = None #'work_dirs/race_cls_fairface_hyperstyle-4race025_v7-top-10-100_10-10_white-0-100-balanced_resnet34_adam/epoch_48.pth'
load_backbone = None  # 'work_dirs/ssp_fakemake-100000_resnet34_adam/epoch_40.pth'
load_from = 'work_dirs/race_cls_fairface_4race025-7000-white-5380-540-balanced_resnet34_adam/epoch_70.pth'