gpus=(0,1,2,3)

data = dict(
    train=dict(
        dataset="cls_fairface",
        root="./datasets/FairFace/images",
        n_classes=[7,],
        face_bbox_txt="./datasets/FairFace/fairface_train_bbox.txt",
        face_label_txt="./datasets/FairFace/fairface_train_race-700-white-280_label.txt",
        size=224,
        samples_per_gpu=16,
        workers_per_gpu=2,
    ),
    val=dict(
        dataset="cls_fairface",
        root="./datasets/FairFace/images",
        size=224,
        n_classes=[7,],
        face_bbox_txt="./datasets/FairFace/fairface_val_bbox.txt",
        face_label_txt="./datasets/FairFace/fairface_race_val_label.txt",
        samples_per_gpu=16,
        workers_per_gpu=4,
    )
)

model = dict(
    name='cls_recognizer',
    backbone=dict(
        name="resnet34"
    ),
    head=dict(
        name="multiclassifier_head",
        n_classes=[7,],
        input_dim=512,
        hidden_dims=[64,]
    )
)

trainer = dict(
    name='cls',
    max_epochs=60,
    save_freq=1,
    print_freq=50,
    val_freq=1
)

optimizer = dict(
    name='adam',
    lr=0.001,
    weight_decay=0.001
)
lr_scheduler = dict(
    name='multisteplrwwarmup',
    milestones=[40, ],
    warmup_epoch=5,
    warmup_lr=0.00001
)

find_unused_parameters = False

resume_from = None