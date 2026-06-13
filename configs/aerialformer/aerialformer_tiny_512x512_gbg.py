_base_ = [
    '../_base_/datasets/gbg_trees_seg.py', '../_base_/models/aerialformer.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule.py'
]

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    # num_classes must equal len(CLASSES) in
    # aerialseg/datasets/gothenburg_tree_seg.py (currently: bg, water, tree).
    #
    # To add a class (e.g. 'building'):
    #   1. Burn it in AF_0_training_data.py with class_value = its CLASSES
    #      index + 1 (the +1 is because reduce_zero_label shifts labels down
    #      by one at load time; 0 stays reserved as "unlabeled"/ignore).
    #   2. Add its name to CLASSES and a color to PALETTE in the dataset class.
    #   3. Increment num_classes here and retrain — checkpoints trained with a
    #      different num_classes cannot be reused (final-layer shape changes).
    decode_head=dict(num_classes=3),
    test_cfg=dict(mode='whole'))

# optimizer
# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=8, workers_per_gpu=8) # 1 GPU x 8 samples/gpu = 8 batch size
