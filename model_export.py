from termcolor import colored
from graph import optimize_graph
from helper import get_run_args, set_logger, import_tf


def get_estimator(args, tf, graph_path):
    from tensorflow.python.estimator.estimator import Estimator
    from tensorflow.python.estimator.run_config import RunConfig
    from tensorflow.python.estimator.model_fn import EstimatorSpec

    def model_fn(features, labels, mode, params):
        with tf.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        input_names = ['input_ids', 'input_mask', 'input_type_ids']

        output = tf.import_graph_def(graph_def,
                                     input_map={k + ':0': features[k] for k in input_names},
                                     return_elements=['final_encodes:0'])

        return EstimatorSpec(mode=mode, predictions={
            'unique_ids ': features['unique_ids'],
            'encodes': output[0]
        })

    config = tf.ConfigProto(device_count={'GPU': 0}, intra_op_parallelism_threads=32, inter_op_parallelism_threads=32)
    config.log_device_placement = False
    config.intra_op_parallelism_threads = 32
    config.inter_op_parallelism_threads = 32
    # session-wise XLA doesn't seem to work on tf 1.10
    # if args.xla:
    #     config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    return Estimator(model_fn=model_fn, config=RunConfig(model_dir=args.checkpoint_dir, session_config=config))


def input_fn_builder():

    def gen():
        for i in range(100):
            yield {
                "unique_ids": [1],
                "input_ids": [[1, 2, 3, 4]],
                "input_mask": [[1, 1, 1, 1]],
                "input_type_ids": [[0, 0, 0, 0]]
            }

    def input_fn():
        return (tf.data.Dataset.from_generator(
            gen,
            output_types={
                "unique_ids": tf.int32,
                "input_ids": tf.int32,
                "input_mask": tf.int32,
                "input_type_ids": tf.int32,
            },
            output_shapes={
                "unique_ids": [None],
                "input_ids": [None, None],
                "input_mask": [None, None],
                "input_type_ids": [None, None],
            }
        ))

    return input_fn

args = get_run_args()

logger = set_logger(colored('VENTILATOR', 'magenta'), args.verbose)
graph_path, bert_config = optimize_graph(args=args)

if graph_path:
    logger.info('optimized graph is stored at: %s' % graph_path)

logger.info('use device %s, load graph from %s' % ('cpu', graph_path))

tf = import_tf(device_id=-1, verbose=args.verbose, use_fp16=args.fp16)
estimator = get_estimator(args=args, tf=tf, graph_path=graph_path)

save_hook = tf.train.CheckpointSaverHook(checkpoint_dir=args.checkpoint_dir, save_secs=1)
predicts = estimator.predict(input_fn=input_fn_builder(), hooks=[save_hook])

for predict in predicts:
    print(predict)

feature_spec = {
    "unique_ids": tf.placeholder(dtype=tf.int32, shape=[None],  name="unique_ids"),
    "input_ids": tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids"),
    "input_mask": tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask"),
    "input_type_ids": tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_type_ids")
}

serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features=feature_spec)

estimator._export_to_tpu = False
estimator.export_saved_model(export_dir_base=args.export_dir, serving_input_receiver_fn=serving_input_fn)