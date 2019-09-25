import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description='post quantization')

parser.add_argument('-saved_model_dir', type=str, required=True,
                    help='directory to saved model')

parser.add_argument('-tflite_file', type=str, required=True,
                    help='path of tflite_file')

args = parser.parse_args()

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=args.saved_model_dir)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
tflite_quant_model = converter.convert()

open(args.tflite_file, "wb").write(tflite_quant_model)