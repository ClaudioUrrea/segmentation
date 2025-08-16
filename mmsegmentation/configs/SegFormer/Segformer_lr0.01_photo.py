crop_size = (512, 512)
data_root = '/automine1d/'
dataset_type = 'automine1d'

default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=2500,
        max_keep_ckpts=-1,
        save_best='mIoU',
        type='CheckpointHook'
    ),
    logger=dict(interval=100, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook')
)

custom_imports = dict(
    imports=[
        'mmseg.models.backbones.mit'
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

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='BiSeNetV1',
        backbone_cfg=dict(
            type='SegFormerB0ForBiSeNet',
            init_cfg=dict(type='Pretrained', checkpoint='checkpoint/mit_b0.pth'
            )
        ),
        context_channels=(512, 1024, 2048),
        spatial_channels=(256, 256, 256, 512),  
        in_channels=3,
        norm_cfg=dict(type='BN', requires_grad=True),
        out_channels=1024,
        out_indices=(0, 1, 2),
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

tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(file_client_args=dict(backend='disk'), type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale_factor=sf, keep_ratio=True) for sf in img_ratios],
            [dict(type='RandomFlip', direction='horizontal', prob=p) for p in [0.0, 1.0]],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')]
        ]
    )
]

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
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
#            dict(type='RandomCLAHE', prob=0.5, clip_limit=3.0, tile_grid_size=(7,7)),
            dict(type='PackSegInputs')
        ]
    ),
    sampler=dict(type='InfiniteSampler', shuffle=True),
    num_workers=2,
    persistent_workers=True
)

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

test_cfg = dict(type='TestLoop')
test_dataloader = val_dataloader
test_evaluator = val_evaluator
test_pipeline = val_dataloader['dataset']['pipeline']

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    name='visualizer',
    vis_backends=vis_backends
)

# === DIRECTORIO DE TRABAJO ===
work_dir = './work_dirs/Segformer/Segformer_lr0.01_photo'
