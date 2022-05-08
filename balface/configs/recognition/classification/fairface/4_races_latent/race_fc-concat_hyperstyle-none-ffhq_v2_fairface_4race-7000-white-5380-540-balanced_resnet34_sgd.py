gpus=(0,1,2,3,5,6,7)

data = dict(
    train=dict(
        dataset="cls_fairface_4races",
        root="./datasets/FairFace/images",
        n_classes=[4,],
        face_bbox_txt="./datasets/FairFace/bbox/fairface_train_bbox.txt",
        face_label_txt="./datasets/FairFace/labels/race/fairface_train_4race-7000-white-5380-540-balanced_label.txt",
        size=224,
        samples_per_gpu=256,
        workers_per_gpu=8,
        mode='train'
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
    name='latentaug_recognizer',
    backbone=dict(
        name="torch_resnet34"
    ),
    latent_encoder=dict(
        name="pretrained_w_encoder"
    ),
    feat_fuser=dict(
        name="fc_concat_fuser",
        in_feat1=512,
        in_feat2=512,
        out_feat=512,
        hidden_dims=(512,)
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
    feat_aug='none',
    aug_num_per_gpu=256,
    aug_ratio=[0.1, 1, 1, 1],
    blend_alpha=0.5
)


trainer = dict(
    name='latentaug',
    max_epochs=70,
    save_freq=2,
    print_freq=10,
    val_freq=2,
    loss_weight='v2',
    sample_ratio=[10, 1, 1, 1.]
)

optimizer = dict(
    name='sgd',
    lr=0.8,
    momentum=0.9,
    weight_decay=0.0001
)

lr_scheduler = dict(
    name='multisteplr',
    milestones=[30, 50],
    # warmup_epoch=2,
    # warmup_lr=0.00001
)

find_unused_parameters = False

resume_from = None #'work_dirs/race_cutmix_fairface_4race-7000-white-5380-540-balanced_resnet34_adam/epoch_34.pth'
load_backbone = None  # 'work_dirs/ssp_fakemake-100000_resnet34_adam/epoch_40.pth'