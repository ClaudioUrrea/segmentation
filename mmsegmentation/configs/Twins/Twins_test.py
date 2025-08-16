crop_size = (512, 512)

default_scope = 'mmseg'

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='BiSeNetV1',
        in_channels=3,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        out_channels=1024,
        out_indices=(0, 1, 2),
        context_channels=(64, 128, 256),
        spatial_channels=(64, 64, 64, 64),
        backbone_cfg=dict(
            type='TwinsPCPVTWrapper',
            out_indices=(0, 1, 2)#,
        )
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
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)
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


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=1.0, direction='horizontal'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]


test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='automine1d',
        data_root='/automine1d',        
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline,
    )
)

test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    output_dir='./work_dirs/Twins/Twins_test',
    prefix='eval'
)

test_cfg = dict(type='TestLoop')

# Necesario para Runner
work_dir = './work_dirs/Twins/Twins_test'

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer'
)


work_dir= './work_dirs/Twins/Twins_test'
