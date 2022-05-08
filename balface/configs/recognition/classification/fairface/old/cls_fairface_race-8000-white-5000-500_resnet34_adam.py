gpus=(0,1,2,3,5,6,7)

data = dict(
    train=dict(
        dataset="cls_fairface",
        root="./datasets/FairFace/images",
        n_classes=[7, 2, 9],
        face_bbox_txt="./datasets/FairFace/bbox/fairface_train_bbox.txt",
        face_label_txt="./datasets/FairFace/labels/race/fairface_train_race-8000-white-5000-500_label.txt",
        size=224,
        samples_per_gpu=256,
        workers_per_gpu=8,
        mode='train'
    ),
    val=dict(
        dataset="cls_fairface",
        root="./datasets/FairFace/images",
        size=224,
        n_classes=[7, 2, 9],
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
        n_classes=[7, 2, 9],
        input_dim=512,
        hidden_dims=[]
    )
)


trainer = dict(
    name='cls',
    max_epochs=36,
    save_freq=1,
    print_freq=10,
    val_freq=4
)

optimizer = dict(
    name='adam',
    lr=0.001,
    weight_decay=0.0005
)

lr_scheduler = dict(
    name='multisteplrwwarmup',
    milestones=[20, 28],
    warmup_epoch=3,
    warmup_lr=0.00001
)

find_unused_parameters = False

resume_from = None
load_backbone = None