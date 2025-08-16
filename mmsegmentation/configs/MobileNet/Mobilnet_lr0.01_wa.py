
crop_size = (512, 512)
data_root = '/automine1d/'
dataset_type = 'automine1d'

custom_imports = dict(
    imports=[

        'mmseg.models.backbones.mobilenet_v2'
    ],
    allow_failed_imports=False
)


default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0)
)

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)

# === MODELO ===

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='BiSeNetV1',
        backbone_cfg=dict(
            type='MobileNetV2_BiSeNetAdapted',
            out_indices=(1, 2, 5, 6),
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            norm_eval=False,
            widen_factor=1.0,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='checkpoint/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'
            ),
            context_channels=(32, 160, 320)
        ),
        context_channels=(32, 160, 320),
        in_channels=3,
        norm_cfg=dict(type='BN', requires_grad=True),
        out_channels=1024,
        out_indices=(0, 1, 2),
        spatial_channels=(64, 64, 64, 128),
        align_corners=False
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=1024,
        channels=1024,
        in_index=0,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        size=(512, 512),
        pad_val=0,
        seg_pad_val=255,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        test_cfg=dict(size_divisor=128)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)



# === TTA DEFINIDO ===
# Agregado: modelo alternativo para validación con test-time augmentation
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(file_client_args=dict(backend='disk'), type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale_factor=sf, keep_ratio=True) for sf in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]],
            [dict(type='RandomFlip', direction='horizontal', prob=p) for p in [0.0, 1.0]],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')]
        ]
    )
]

# === OPTIMIZADOR ===
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
)
param_scheduler = [
    dict(begin=0, by_epoch=False, end=1000, start_factor=0.1, type='LinearLR'),
    dict(begin=1000, by_epoch=False, end=160000, eta_min=0.0001, power=0.9, type='PolyLR')
]
randomness = dict(seed=0)
resume = False

# === ENTRENAMIENTO ===
train_cfg = dict(max_iters=40000, type='IterBasedTrainLoop', val_interval=500)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type='automine1d',
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='RandomResize', scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
#            dict(type='PhotoMetricDistortion'),
#            dict(
#                type='RandomCLAHE',
#                prob=0.5, clip_limit=3.0, tile_grid_size=(7,7)
#            ),
            dict(type='PackSegInputs')
        ]
    ),
    sampler=dict(type='InfiniteSampler', shuffle=True),
    num_workers=2,
    persistent_workers=True
)

# === VALIDACIÓN ===
# Modificado: activamos validación usando TTA
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='automine1d',
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    num_workers=4,
    persistent_workers=True
)
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])

# === TEST ===
test_cfg = dict(type='TestLoop')
test_dataloader = val_dataloader
test_evaluator = val_evaluator
test_pipeline = val_dataloader['dataset']['pipeline']

# === VISUALIZACIÓN ===
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    name='visualizer',
    vis_backends=vis_backends
)

# === DIRECTORIO DE TRABAJO ===
work_dir = './work_dirs/Mobilnet/Mobilnet_lr0.01_wa'
