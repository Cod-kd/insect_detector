import tensorflow as tf
interp = tf.lite.Interpreter("models/model.tflite")
interp.allocate_tensors()
# details = interp.get_tensor_details()
# vagy:
for d in interp._get_ops_details(): print(d['op_name'])
