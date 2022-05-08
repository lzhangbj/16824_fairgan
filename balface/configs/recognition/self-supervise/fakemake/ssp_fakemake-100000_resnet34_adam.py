gpus=(0,1,2,3,4,5,6,7)

data = dict(
    train=dict(
        dataset="ssp_base",
        root="./datasets/FakeMake/images",
        face_bbox_txt="./datasets/FakeMake/bbox/fakemake-100000_bbox.txt",
        samples_per_gpu=256,
        workers_per_gpu=8,
        size=224
    )
)

model = dict(
    name='ssp_recognizer',
    backbone=dict(
        name="torch_resnet34"),
    head=dict(
        name="contrastive_head",
        embed_dim=128,
        input_dim=512,
        hidden_dims=[256,],
        T=0.07)
)

trainer = dict(
    name="ssp",
    max_epochs=50,
    save_freq=5,
    print_freq=10
)

optimizer = dict(
    name='adam',
    lr=0.001,
    # momentum=0.9,
    weight_decay=0.0005
)

lr_scheduler = dict(
    name='multisteplrwwarmup',
    milestones=[30, 40],
    warmup_epoch=3,
    warmup_lr=0.00001
)

resume_from = None