#!/usr/bin/env python
# Usage: python3.8 reproduce_baseline.py input_dir output_dir ingestion_program_dir submission_program_dir

# Use default location for the input and output data:
# If no arguments to run.py are provided, this is where the data will be found
# and the results written to. Change the root_dir to your local directory.
root_dir = "/app/"
default_input_dir = root_dir + "data"
default_output_dir = root_dir + "output"
default_program_dir = root_dir + "program"
default_submission_dir = root_dir + "ingested_program"


# General purpose functions
import time
overall_start = time.time()         # <== Mark starting time
import os
from sys import argv, path
import sys
import datetime
import json
import importlib
the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
import numpy as np
# =========================== BEGIN PROGRAM ================================

class TimeoutException(Exception):
    """timeoutexception"""

class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

# =========================== BEGIN MAIN ================================
if __name__=="__main__" :

    # =========================== HOUSEKEEPING ================================

    #### Check whether everything went well (no time exceeded)
    execution_success = True

    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
        program_dir= default_program_dir
        submission_dir= default_submission_dir
    else:
        input_dir = os.path.abspath(argv[1])
        output_dir = os.path.abspath(argv[2])
        program_dir = os.path.abspath(argv[3])
        submission_dir = os.path.abspath(argv[4])

    print("Using input_dir: " + input_dir)
    print("Using output_dir: " + output_dir)
    print("Using program_dir: " + program_dir)
    print("Using submission_dir: " + submission_dir)

	# Our libraries
    path.append (program_dir)
    path.append (submission_dir)

    print("## Starting Ingestion program ##")

 

    # =========================== START BASELINE 1 ================================
    start_total_time = time.time()
    from lips import get_root_path
    from lips.benchmark.powergridBenchmark import PowerGridBenchmark

    LIPS_PATH = get_root_path()
    BENCH_CONFIG_PATH = os.path.join(LIPS_PATH, "config", "benchmarks", "l2rpn_case14_sandbox.ini")
    DATA_PATH = '/app/data/lips_case14_sandbox/'
    TRAINED_MODELS = os.path.join(submission_dir, "trained_models")
    LOG_PATH = "logs.log"

   # =========================== LOAD BENCHMARK ================================
    
    benchmark_kwargs = {"attr_x": ("prod_p", "prod_v", "load_p", "load_q"),
                        "attr_y": ("a_or", "a_ex", "p_or", "p_ex", "v_or", "v_ex"),
                        "attr_tau": ("line_status", "topo_vect"),
                        "attr_physics": None}

    benchmark = PowerGridBenchmark(benchmark_path=DATA_PATH,
                                config_path=BENCH_CONFIG_PATH,
                                benchmark_name="Benchmark_competition",
                                load_data_set=True, 
                                load_ybus_as_sparse=False,
                                log_path=LOG_PATH,
                                **benchmark_kwargs)
    

    # =========================== LOAD MODEL ================================
    
    import tensorflow as tf

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    memory_limit = 20000

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
        except RuntimeError as e:
            print(e)

    from lips.augmented_simulators.tensorflow_models import TfFullyConnected
    from lips.dataset.scaler import StandardScaler

    SIM_CONFIG_PATH = os.path.join(submission_dir, "tf_fc.ini")

    tf_fc = TfFullyConnected(name="tf_fc",
                         bench_config_path=BENCH_CONFIG_PATH,
                         bench_config_name="Benchmark_competition",
                         bench_kwargs=benchmark_kwargs,
                         sim_config_path=SIM_CONFIG_PATH,
                         sim_config_name="DEFAULT",
                         scaler=StandardScaler,
                         log_path=LOG_PATH)
    
    print("MODEL PARAMETERS ", tf_fc.params)


    # =========================== TRAIN MODEL ================================
    tf_fc.train(train_dataset=benchmark.train_dataset,
            val_dataset=benchmark.val_dataset,
            epochs=1
           )
    
    # =========================== SAVE MODEL ================================
    SAVE_PATH = os.path.join(TRAINED_MODELS, "fully_connected")
    tf_fc.save(SAVE_PATH)   

    print("SUMMARY ")
    print( tf_fc.summary())

    # =========================== Evaluate metrics ================================

    tf_fc_metrics = benchmark.evaluate_simulator(augmented_simulator=tf_fc,
                                             eval_batch_size=128,
                                             dataset="all",
                                             shuffle=False,
                                             save_path=submission_dir,
                                             save_predictions=output_dir
                                            )

    print("METRICS ", tf_fc_metrics)
    json_metrics = json.dumps(tf_fc_metrics, indent=4, cls=NumpyFloatValuesEncoder)
    
    with open(os.path.join(output_dir, 'json_metrics.json'), "w") as outfile:
        outfile.write(json_metrics)