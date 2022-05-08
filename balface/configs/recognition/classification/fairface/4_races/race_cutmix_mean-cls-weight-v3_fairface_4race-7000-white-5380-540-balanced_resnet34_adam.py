gpus=(0,1,2,3,5,6,7)

data = dict(
    train=dict(
        dataset="cutmix_fairface_4races",
        root="./datasets/FairFace/images",
        n_classes=[4,],
        face_bbox_txt="./datasets/FairFace/bbox/fairface_train_bbox.txt",
        face_label_txt="./datasets/FairFace/labels/race/fairface_train_4race-7000-white-5380-540-balanced_label.txt",
        size=224,
        samples_per_gpu=256,
        workers_per_gpu=8,
        mode='train',
        aug_num=7000
    ),
    val=dict(
        dataset="cls_fairface_4races",
        root="./datasets/FairFace/images",
        size=224,
        n_classes=[4,],
        face_bbox_txt="./datasets/FairFace/bbox/fairface_val_bbox.txt",
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
    use_weight=True
)


trainer = dict(
    name='cutmix',
    max_epochs=70,
    save_freq=2,
    print_freq=10,
    val_freq=2,
    loss_weight='v3',
    sample_ratio=[10, 1, 1, 1.]
)

optimizer = dict(
    name='adam',
    lr=0.0003,
    weight_decay=0.0001
)

lr_scheduler = dict(
    name='multisteplr',
    milestones=[30, 50],
    # warmup_epoch=2,
    # warmup_lr=0.00001
)

find_unused_parameters = False

resume_from = None # 'work_dirs/race_cutmix_mean-cls-weight-v2_fairface_4race-7000-white-5380-540-balanced_resnet34_adam/epoch_70.pth'
load_backbone = None  # 'work_dirs/ssp_fakemake-100000_resnet34_adam/epoch_40.pth'