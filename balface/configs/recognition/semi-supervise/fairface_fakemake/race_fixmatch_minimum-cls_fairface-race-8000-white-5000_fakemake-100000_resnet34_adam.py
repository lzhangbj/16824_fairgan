gpus=(0,1,2,3,5,6,7)

data = dict(
    labeled_train=dict(
        dataset="cls_fairface",
        root="./datasets/FairFace/images",
        n_classes=[7,],
        size=224,
        face_bbox_txt="./datasets/FairFace/bbox/fairface_train_bbox.txt",
        face_label_txt="./datasets/FairFace/labels/race/fairface_train_race-8000-white-5000-500_label.txt",
        samples_per_gpu=128,
        workers_per_gpu=4,
        mode='train'
    ),
    unlabeled_train=dict(
        dataset="fixmatch",
        root="./datasets/FakeMake/images",
        size=224,
        face_bbox_txt="./datasets/FakeMake/bbox/fakemake-100000_bbox.txt",
        samples_per_gpu=128,
        workers_per_gpu=4,
        mode='train'
    ),
    val=dict(
        dataset="cls_fairface",
        root="./datasets/FairFace/images",
        size=224,
        n_classes=[7,],
        face_bbox_txt="./datasets/FairFace/bbox/fairface_train_bbox.txt",
        face_label_txt="./datasets/FairFace/labels/all/fairface_train_all_label.txt",
        samples_per_gpu=128,
        workers_per_gpu=4,
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
        n_classes=[7, ],
        input_dim=512,
        hidden_dims=[]
    )
)

trainer = dict(
    name='fixmatch',
    max_epochs=40,
    save_freq=2,
    print_freq=10,
    val_freq=2,
    ssl_weight=0.5,
    resample='minimum_cls'
)

optimizer = dict(
    name='adam',
    lr=0.0005,
    weight_decay=0.0005
)

lr_scheduler = dict(
    name='multisteplrwwarmup',
    milestones=[24, 34],
    warmup_epoch=2,
    warmup_lr=0.00001
)


find_unused_parameters = False

resume_from = 'work_dirs/race_fixmatch_minimum-cls_fairface-race-8000-white-5000_fakemake-100000_resnet34_adam/epoch_40.pth'
load_backbone = None # 'work_dirs/race_cls_fairface_race-8000-white-5000-500_resnet34_adam/epoch_40.pth'
load_model = None