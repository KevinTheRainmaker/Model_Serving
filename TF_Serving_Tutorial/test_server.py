import tensorflow as tf
import numpy as np

import tempfile
import os

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

history = model.fit(xs, ys, epochs=500, verbose=0)

print('model learning done')

print(model.predict([10.0]))

MODEL_DIR = tempfile.gettempdir()
# if you use colab, un-commnet underline
# os.environ['MODEL_DIR'] = MODEL_DIR

version = 1
export_path = os.path.join(MODEL_DIR, str(version))

if os.path.isdir(export_path):
    print('Remove other models')
    os.system(f'rm -r {export_path}')

print(export_path)

model.save(export_path, save_format='tf')

print(f'export_path = {export_path}')
os.system(f'ls -l {export_path}')

# os.system(f'nohup tensorflow_model_server --rest_api_port=8501 --model_name=helloworld --model_base_path="{MODEL_DIR}" > server.log 2>&1 &')
os.system(f"nohup tensorflow_model_server --rest_api_port=8501 \
--model_config_file=model.config \
--model_config_file_poll_wait_seconds=60 > server.log 2>&1 &")