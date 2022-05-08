gpus=(4,5,6,7)

data = dict(
    train=dict(
        dataset="ssp_base",
        root="./datasets/FakeMake/images",
        face_bbox_txt="./datasets/FakeMake/fakemake-10000_bbox.txt",
        samples_per_gpu=128,
        workers_per_gpu=8,
        size=224
    )
)

model = dict(
    name='ssp_recognizer',
    backbone=dict(
        name="resnet34"),
    head=dict(
        name="contrastive_head",
        embed_dim=128,
        input_dim=512,
        hidden_dims=[256,],
        T=0.07)
)

trainer = dict(
    name="ssp",
    max_epochs=40,
    save_freq=1,
    print_freq=50
)

optimizer = dict(
    name='adam',
    lr=0.001,
    # momentum=0.9,
    weight_decay=0.0005
)

lr_scheduler = dict(
    name='multisteplrwwarmup',
    milestones=[30, ],
    warmup_epoch=2,
    warmup_lr=0.00001
)

resume_from = None