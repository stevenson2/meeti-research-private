#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R13 PTB-XL single-lead smoke builders.

Wraps the electron-project builder (read-only location) so R13 can construct
a 5-class / 2-class single-lead model without copying or editing that file.
"""
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, Model

ELECTRON = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
                'ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(ELECTRON / 'pc_tools/ecg_dl'))

from models.resnet_lite_1d import build_ecg_resnet_lite  # noqa: E402

LARGE_CFG = dict(
    filters=(16, 32, 64, 128),
    blocks_per_stage=(2, 3, 3, 1),
    kernel_sizes=(7, 5, 3, 3),
    strides=(1, 2, 2, 1),
    dropout_rate=0.4,
)


def build_ptbxl_large(input_shape=(1000, 1), n_classes=2,
                      output_activation='softmax', name='ptbxl_resnet_lite'):
    """ECG-ResNet-Lite 'large' config with arbitrary output classes.

    The electron builder always ends with softmax; for multi-label 5-class
    PTB-XL experiments we rebuild only the head with sigmoid while reusing
    the shared trunk layers.
    """
    base = build_ecg_resnet_lite(
        input_shape=input_shape,
        n_classes=n_classes,
        **LARGE_CFG,
    )
    if output_activation == 'softmax':
        base._name = name
        return base

    inp = base.input
    x = base.get_layer('fc1').output
    x = base.get_layer('do')(x)
    out = layers.Dense(n_classes, activation=output_activation,
                       name='out_multi')(x)
    model = Model(inp, out, name=name)
    return model
