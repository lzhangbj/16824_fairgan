gpus=(0,1,2,3,5,6,7)

data = dict(
    labeled_train=dict(
        dataset="cls_fairface",
        root="./datasets/FairFace/images",
        n_classes=[7, 2, 9],
        size=224,
        face_bbox_txt="./datasets/FairFace/bbox/fairface_train_bbox.txt",
        face_label_txt="./datasets/FairFace/labels/all/fairface_train_all_label.txt",
        samples_per_gpu=256,
        workers_per_gpu=4,
        mode='train'
    ),
    unlabeled_train=dict(
        dataset="unlabel",
        root="./datasets/FakeMake/images",
        size=224,
        face_bbox_txt="./datasets/FakeMake/bbox/fakemake-100000_bbox.txt",
        samples_per_gpu=256,
        workers_per_gpu=4,
        mode='train'
    ),
    val=dict(
        dataset="cls_fairface",
        root="./datasets/FairFace/images",
        size=224,
        n_classes=[7, 2, 9],
        face_bbox_txt="./datasets/FairFace/bbox/fairface_val_bbox.txt",
        face_label_txt="./datasets/FairFace/labels/all/fairface_val_all_label.txt",
        samples_per_gpu=128,
        workers_per_gpu=4,
        mode='val'
    )
)

model = dict(
    name="ssl_recognizer",
    backbone=dict(
        name="torch_resnet34",
    ),
    head=dict(
        name="multiclassifier_head",
        n_classes=[7, 2, 9],
        input_dim=512,
        hidden_dims=[]
    )
)

trainer = dict(
    name='ssl',
    max_epochs=20,
    save_freq=1,
    print_freq=10,
    val_freq=4,
    ssl_weight=0.1,
    pseudo_label_cycle=1

)

optimizer = dict(
    name='adam',
    lr=0.001,
    weight_decay=0.0005
)

lr_scheduler = dict(
    name='multisteplrwwarmup',
    milestones=[12, 16],
    warmup_epoch=3,
    warmup_lr=0.00001
)


find_unused_parameters = False

resume_from = None
load_backbone = None
load_model = None