import onnxruntime as rt



sess = rt.InferenceSession('./onnx_checkpoints/resnet50.onnx')

input_name = sess.get_inputs()[0].name
print("input name", input_name)
input_shape = sess.get_inputs()[0].shape
print("input shape", input_shape)
input_type = sess.get_inputs()[0].type
print("input type", input_type)

print()

input_name = sess.get_outputs()[0].name
print("output name", input_name)
input_shape = sess.get_outputs()[0].shape
print("output shape", input_shape)
input_type = sess.get_outputs()[0].type
print("output type", input_type)