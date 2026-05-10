import tensorflow as tf
interp = tf.lite.Interpreter("models/model.tflite")
interp.allocate_tensors()
for d in interp._get_ops_details(): print(d['op_name'])
