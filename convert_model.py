#!/usr/bin/env python3
"""
Convert asl_model.keras → TensorFlow.js format.

If tensorflowjs import fails due to protobuf conflicts, run with:

    pip install "protobuf>=6.31.1"
    python convert_model.py

Output is written to: frontend/public/model/
"""

import json, os

MODEL_PATH   = './asl_model.keras'
CLASSES_PATH = './asl_classes.json'
OUT_DIR      = './frontend/public/model'

os.makedirs(OUT_DIR, exist_ok=True)

import tensorflow as tf
import tensorflowjs as tfjs

print('Loading model...')
model = tf.keras.models.load_model(MODEL_PATH)

print('Converting to TF.js format...')
tfjs.converters.save_keras_model(model, OUT_DIR)

with open(CLASSES_PATH) as f:
    classes = json.load(f)
with open(os.path.join(OUT_DIR, 'classes.json'), 'w') as f:
    json.dump(classes, f)

print(f'\nDone! Files saved to {OUT_DIR}/')
print('Commit the frontend/public/model/ folder to git before deploying.')
