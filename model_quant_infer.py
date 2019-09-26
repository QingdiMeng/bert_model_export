import tensorflow as tf
import time

interpreter = tf.lite.Interpreter(model_path="./bert_lite.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

input_ids = [[
101,
2769,
812,
4638,
686,
4518,
3221,
784,
720,
102,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0
]]

input_mask = [[
1,
1,
1,
1,
1,
1,
1,
1,
1,
1,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0
]]

input_types = [[
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0,
0
]]

for i in range(0, 10):
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_ids)
    interpreter.set_tensor(input_details[1]['index'], input_mask)
    interpreter.set_tensor(input_details[2]['index'], input_types)
    interpreter.set_tensor(input_details[3]['index'], [i])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # print(output_data)
    print(time.time() - start)



