# 33.030, 0.610

gpus=(0,1,2,3,5,6,7)

data = dict(
    train=dict(
        dataset="imbalance_cifar_100",
        root='./datasets',
        n_class=100,
        imb_factor=0.02,
        samples_per_gpu=1024,
        workers_per_gpu=8,
        mode='train'
    ),
    val=dict(
        dataset="cifar_100",
        root='./datasets',
        n_class=100,
        samples_per_gpu=256,
        workers_per_gpu=8,
        mode='val'
    )
)

model = dict(
    name='benchmark_recognizer',
    backbone=dict(
        name="resnet32"
    ),
    head=dict(
        name="classifier_head",
        n_class=100,
        input_dim=64,
        hidden_dims=[],
        loss='ce',
        norm_weights=False
    ),
    use_weight=True
)


trainer = dict(
    name='cifar',
    max_epochs=200,
    save_freq=5,
    print_freq=10,
    val_freq=5,
    loss_weight='v3',
    expected_ratio=0.9,
    meta_epoch=100
)

optimizer = dict(
    name='sgd',
    lr=0.8,
    momentum=0.9,
    weight_decay=2e-4
)

find_unused_parameters = False

resume_from = None # 'work_dirs/race_cls_mean-cls-weight_fairface_race-8000-white-5000-500-balanced_resnet34_adam/epoch_40.pth'
load_backbone = None  # 'work_dirs/ssp_fakemake-100000_resnet34_adam/epoch_40.pth'