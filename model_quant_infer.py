import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="./bert_lite.tfile")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

interpreter.set_tensor(input_details[0]['index'], [
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
])

interpreter.set_tensor(input_details[1]['index'], [
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
])

interpreter.set_tensor(input_details[2]['index'], [
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
])

interpreter.set_tensor(input_details[3]['index'], 1)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)



