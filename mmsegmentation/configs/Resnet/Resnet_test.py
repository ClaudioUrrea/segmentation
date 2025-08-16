crop_size = (512, 512)

default_scope = 'mmseg'
model = dict(
    backbone=dict(
        align_corners=False,
        backbone_cfg=dict(
            contract_dilation=True,
            depth=50,
            dilations=(
                1,
                1,
                1,
                1,
            ),
            in_channels=3,

            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            norm_eval=False,
            num_stages=4,
            out_indices=(
                0,
                1,
                2,
                3,
            ),
            strides=(
                1,
                2,
                2,
                2,
            ),
            style='pytorch',
            type='ResNet'),
        context_channels=(
            512,
            1024,
            2048,
        ),
        in_channels=3,
        init_cfg=None,
        norm_cfg=dict(requires_grad=True, type='BN'),
        out_channels=1024,
        out_indices=(
            0,
            1,
            2,
        ),
        spatial_channels=(
            256,
            256,
            256,
            512,
        ),
        type='BiSeNetV1'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        test_cfg=dict(size_divisor=128),
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=1024,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=1024,
        in_index=0,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=2,
        num_convs=1,
        type='FCNHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')

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
    output_dir='./work_dirs/Resnet/Resnet_test',
    prefix='eval'
)

test_cfg = dict(type='TestLoop')

# Necesario para Runner
work_dir = './work_dirs/Resnet/Resnet_test'

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer'
)


work_dir= './work_dirs/Resnet/Resnet_test'
