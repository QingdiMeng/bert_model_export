from termcolor import colored
from graph import optimize_graph
from helper import get_run_args, set_logger, import_tf


def get_estimator(tf, graph_path):
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

    config = tf.ConfigProto(device_count={'GPU': 0})
    config.log_device_placement = False
    # session-wise XLA doesn't seem to work on tf 1.10
    # if args.xla:
    #     config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    return Estimator(model_fn=model_fn, config=RunConfig(session_config=config))


def predict_input_fn():
    return {
        "unique_ids": tf.constant([1], dtype=tf.int32),
        "input_ids": tf.constant([[1,2,3,4]], dtype=tf.int32),
        "input_mask": tf.constant([[1,1,1,1]], dtype=tf.int32),
        "input_type_ids": tf.constant([0,0,0,0], dtype=tf.int32),
    }

args = get_run_args()

logger = set_logger(colored('VENTILATOR', 'magenta'), args.verbose)
graph_path, bert_config = optimize_graph(args=args)

if graph_path:
    logger.info('optimized graph is stored at: %s' % graph_path)

logger.info('use device %s, load graph from %s' % ('cpu', graph_path))

tf = import_tf(device_id=-1, verbose=args.verbose, use_fp16=args.fp16)
estimator = get_estimator(tf=tf, graph_path=graph_path)

save_hook = tf.train.CheckpointSaverHook(args.export_dir, save_secs=1)
estimator.predict(input_fn=predict_input_fn, hooks=[save_hook])

feature_spec = {
    "unique_ids": tf.placeholder(dtype=tf.int32, shape=[None],  name="unique_ids"),
    "input_ids": tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids"),
    "input_mask": tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask"),
    "input_type_ids": tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_type_ids")
}

serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features=feature_spec)

estimator._export_to_tpu = False
estimator.export_saved_model(export_dir_base=args.export_dir, serving_input_receiver_fn=serving_input_fn)