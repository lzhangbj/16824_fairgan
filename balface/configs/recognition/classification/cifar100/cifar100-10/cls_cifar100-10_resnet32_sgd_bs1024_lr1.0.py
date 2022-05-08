gpus=(0,1,2,3,5,6,7)

data = dict(
    train=dict(
        dataset="imbalance_cifar_100",
        root='./datasets',
        n_class=100,
        imb_factor=0.1,
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
    use_weight=False
)


trainer = dict(
    name='cifar',
    max_epochs=200,
    save_freq=5,
    print_freq=10,
    val_freq=5,
    loss_weight='none'
)

optimizer = dict(
    name='sgd',
    lr=1.0,
    momentum=0.9,
    weight_decay=2e-4
)

find_unused_parameters = False

resume_from = None # 'work_dirs/cls_cifar100-10_resnet32_sgd/epoch_200.pth'
load_backbone = None  # 'work_dirs/ssp_fakemake-100000_resnet34_adam/epoch_40.pth'