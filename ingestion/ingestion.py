#!/usr/bin/env python

# Usage: python $ingestion_program/ingestion.py $input $output $ingestion_program $submission_program

# =========================== BEGIN OPTIONS ==============================
# Verbose mode:
##############
# Recommended to keep verbose = True: shows various progression messages
verbose = True # outputs messages to stdout and stderr for debug purposes



# =============================================================================
# =========================== END USER OPTIONS ================================
# =============================================================================

# General purpose functions
import time
overall_start = time.time()         # <== Mark starting time
import os
from sys import argv, path
import argparse
import sys
import datetime
import json
import importlib
import tensorflow as tf
from lips.config import ConfigManager
from lips.benchmark.powergridBenchmark import PowerGridBenchmark
from lips.dataset.utils.powergrid_utils import get_kwargs_simulator_scenario
from lips.benchmark.powergridBenchmark import get_env
from lips.evaluation.powergrid_evaluation import PowerGridEvaluation
from pprint import pprint

# Use default location for the input and output data:
# If no arguments to run.py are provided, this is where the data will be found
# and the results written to. Change the root_dir to your local directory.
root_dir = os.path.abspath(os.path.curdir)
default_input_dir = root_dir + "/input_data_local"
default_output_dir = root_dir + "/submission_results"
default_submission_dir = root_dir + "/competition_submissions"
print(default_input_dir)

the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

# =========================== BEGIN PROGRAM ================================
def fileExists(path):
    if not os.path.exists(path):
        print(path)
        raise ModelApiError("Missing file : ", path)
        

def import_parameters(submission_dir):
    ## import parameters.json as a dictionary
    path_submission_parameters = os.path.join(submission_dir, 'parameters.json')
    if not os.path.exists(path_submission_parameters):
        raise ModelApiError("Missing parameters.json file")
    with open(os.path.join(submission_dir, 'parameters.json')) as json_file:
        parameters = json.load(json_file)
    return parameters

def exit_program():
    print("Error exiting")
    sys.exit(0)

class ModelApiError(Exception):
    """Model api error"""

    def __init__(self, msg=""):
        self.msg = msg
        print(msg)
        exit_program()  


class TimeoutException(Exception):
    """timeoutexception"""

def create_parser():
    this_parser = argparse.ArgumentParser()
    this_parser.add_argument("--input_dir", type=str, default=default_input_dir,
                             help="the input directory where the data could be found")
    this_parser.add_argument("--submission_dir", type=str, default=default_submission_dir,
                             help="the directory containing a valid submission from competition")
    this_parser.add_argument("--submission_name", type=str,
                             help="the name of submission that should be executed here")
    this_parser.add_argument("--output_dir", type=str, default=default_output_dir,
                             help="output directory where the results of evaluation should be saved")
    this_parser.add_argument("--nb_run", type=int, default=3,
                             help="Number of runs for a submission")
    this_parser.add_argument("--device", type=str, default="cuda:0",
                             help="device on which the submissions shoud be executed")
    
    return this_parser

if __name__=="__main__" :
    #### Check whether everything went well (no time exceeded)
    execution_success = True
    
    parser = create_parser()
    args = parser.parse_args()
    command = ' '.join([sys.executable] + sys.argv)
    
    input_dir = str(args.input_dir)
    submission_dir = str(args.submission_dir)
    submission_name = str(args.submission_name)
    submission_dir = os.path.join(submission_dir, submission_name)
    output_dir = str(args.output_dir)
    output_dir = os.path.join(output_dir, submission_name)
    nb_run = int(args.nb_run)
    device = str(args.device)

    if verbose:
        print("Using input_dir: " + input_dir)
        print("Using output_dir: " + output_dir)
        print("Using submission_dir: " + submission_dir)

	# Our libraries
    path.append(submission_dir)

    #### Test with LIPS dataset ####
    print("## Starting Ingestion program ##")

    # import configuration file 
    run_parameters = import_parameters(submission_dir)
    print("Run parameters: ", run_parameters)


    start_total_time = time.time()
    from lips import get_root_path

    LIPS_PATH = get_root_path()
    BENCH_CONFIG_PATH = os.path.join(root_dir, "configs", "benchmarks", "lips_idf_2023.ini")
    DATA_PATH = os.path.join(root_dir, "input_data_local","lips_idf_2023")
    LOG_PATH = "logs.log"

    TRAINED_MODELS = os.path.join(submission_dir, "trained_models")
    SIM_CONFIG_PATH = os.path.join(submission_dir,"config.ini")
    
    config = ConfigManager(path=SIM_CONFIG_PATH, 
                           section_name=run_parameters["simulator_config"]["config_name"])
    
    # FIXME: if evaluateonly true : copy results for evaluation, deactivate option for competition phase



    benchmark = PowerGridBenchmark(benchmark_name="Benchmark_competition",
                                   benchmark_path=DATA_PATH,
                                   load_data_set=True,
                                   log_path=None,
                                   config_path=BENCH_CONFIG_PATH,
                                   load_ybus_as_sparse=True,
                                   **config.get_option("benchmark_kwargs")
                                   )

    print("Input attributes (features): ", benchmark.config.get_option("attr_x"))
    print("Output attributes (targets): ", benchmark.config.get_option("attr_y"))

    simulator_parameters = run_parameters["simulator_config"]


    print("Preparing scaler")
    if simulator_parameters["scaler_type"] == "simple":
        print("Loading LIPS scaler " + simulator_parameters["scaler"])
        scaler_module = importlib.import_module("lips.dataset.scaler."+simulator_parameters["scaler_module"])
        scaler_class = getattr(scaler_module, simulator_parameters["scaler"])
    elif simulator_parameters["scaler_type"] == "custom":
        print("Custom scaler")
        print("Loading custom scaler from submission directory")
        fileExists(os.path.join(submission_dir,simulator_parameters["scaler_file"]+".py"))
        ## load custom scaler from submission directory
        scaler_module = importlib.import_module(simulator_parameters["scaler_file"])
        scaler_class = getattr(scaler_module, simulator_parameters["scaler"])
    else:
        print("No scaler specified")
        scaler_class = None

    for run_ in range(nb_run):
        print(f"**************RUN #{run_}*****************")
        date_now = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
        print("Preparing Simulator")
        if simulator_parameters["simulator_type"] == "simple_torch":
            print("Loading LIPS torch simulator " + simulator_parameters["model"])
            simulator_module = importlib.import_module("lips.augmented_simulators.torch_models."+simulator_parameters["model_type"])
            simulator_class = getattr(simulator_module, simulator_parameters["model"])

            from lips.augmented_simulators.torch_simulator import TorchSimulator
            simulator = TorchSimulator(name=simulator_parameters["name"],
                            model=simulator_class,
                            scaler=scaler_class,
                            log_path="log_benchmark",
                            device=device,
                            seed=simulator_parameters["seed"],
                            bench_config_path=BENCH_CONFIG_PATH,
                            bench_config_name="Benchmark_competition",
                            bench_kwargs=config.get_option("benchmark_kwargs"),
                            sim_config_path=SIM_CONFIG_PATH,
                            sim_config_name=simulator_parameters["config_name"],
                            **run_parameters["simulator_extra_parameters"]                      
                            )

        elif simulator_parameters["simulator_type"] == "intermediate_torch": 
            print("Custom torch LIPS simulator")
            print("Loading custom simulator from submission directory")
            ## load custom simulator from submission directory
            fileExists(os.path.join(submission_dir,simulator_parameters["simulator_file"]+'.py'))
            # Import user-provided simulator code
            simulator_module = importlib.import_module(simulator_parameters["simulator_file"])
            simulator_class = getattr(simulator_module, simulator_parameters["model"])
            from lips.augmented_simulators.torch_simulator import TorchSimulator
            simulator = TorchSimulator(name=simulator_parameters["name"],
                            model=simulator_class,
                            scaler=scaler_class,
                            log_path="log_benchmark",
                            device=device,
                            seed=simulator_parameters["seed"],
                            bench_config_path=BENCH_CONFIG_PATH,
                            bench_config_name='Benchmark_competition',
                            bench_kwargs=config.get_option("benchmark_kwargs"),
                            sim_config_path=SIM_CONFIG_PATH,
                            sim_config_name=simulator_parameters["config_name"],
                            **run_parameters["simulator_extra_parameters"]                      
                            )

        elif simulator_parameters["simulator_type"] == "simple_tf":
            print("Loading LIPS torch simulator " + simulator_parameters["model"])
            if simulator_parameters["model_type"] == "leap_net":
                simulator_module = importlib.import_module("lips.augmented_simulators.tensorflow_models.powergrid."+simulator_parameters["model_type"])
                topo_vect_to_tau = run_parameters["simulator_extra_parameters"].get("topo_vect_to_tau")
                if topo_vect_to_tau is not None:
                    if topo_vect_to_tau == "given_list":
                        topo_actions = benchmark.config.get_option("dataset_create_params")["reference_args"]["topo_actions"]
                        kwargs_tau = []
                        for el in topo_actions:
                            kwargs_tau.append(el["set_bus"]["substations_id"][0])
                        run_parameters["simulator_extra_parameters"]["kwargs_tau"] = kwargs_tau                
            else:
                simulator_module = importlib.import_module("lips.augmented_simulators.tensorflow_models."+simulator_parameters["model_type"])
            simulator_class = getattr(simulator_module, simulator_parameters["model"])

            activation = run_parameters["simulator_extra_parameters"].get("activation")
            if activation is not None:
                if activation == "leaky_relu":
                    run_parameters["simulator_extra_parameters"]["activation"] = tf.keras.layers.LeakyReLU(alpha=0.01)            
                
            simulator = simulator_class(name=simulator_parameters["name"],
                                        bench_config_path=BENCH_CONFIG_PATH,
                                        bench_config_name="Benchmark_competition",
                                        bench_kwargs=config.get_option("benchmark_kwargs"),
                                        sim_config_path=SIM_CONFIG_PATH,
                                        sim_config_name=simulator_parameters["config_name"],
                                        scaler=scaler_class,
                                        log_path=LOG_PATH,
                                        **run_parameters["simulator_extra_parameters"])
            
        elif simulator_parameters["simulator_type"] == "intermediate_tf": 
            print("Custom TF LIPS simulator")
            print("Loading custom simulator from submission directory")
            ## load custom simulator from submission directory
            fileExists(os.path.join(submission_dir,simulator_parameters["simulator_file"]+'.py'))
            # Import user-provided simulator code
            simulator_module = importlib.import_module(simulator_parameters["simulator_file"])
            simulator_class = getattr(simulator_module, simulator_parameters["model"])

            simulator = simulator_class(name=simulator_parameters["name"],
                                        bench_config_path=BENCH_CONFIG_PATH,
                                        bench_config_name="Benchmark_competition",
                                        bench_kwargs=config.get_option("benchmark_kwargs"),
                                        sim_config_path=SIM_CONFIG_PATH,
                                        sim_config_name=simulator_parameters["config_name"],
                                        scaler=scaler_class,
                                        log_path=LOG_PATH,
                                        **run_parameters["simulator_extra_parameters"])    
                

        elif simulator_parameters["simulator_type"] == "advanced":
            # print("Loading custom simulator " + simulator_parameters["model"])
            print("Loading custom simulator " + simulator_parameters["simulator_file"])
            ## load custom simulator from submission directory
            fileExists(os.path.join(submission_dir,simulator_parameters["simulator_file"]+'.py'))
            # Import user-provided simulator code
            simulator_module = importlib.import_module(simulator_parameters["simulator_file"])
            simulator_class = getattr(simulator_module, simulator_parameters["simulator"])
            # simulator = simulator_class(benchmark=benchmark,
            #                             device="cuda:0",
            #                             **run_parameters["simulator_extra_parameters"]
            #                             )
            # model = model_class(config)
            simulator = simulator_class(benchmark,
                                        config=config, 
                                        scaler=scaler_class,
                                        device=device,
                                        **run_parameters["simulator_extra_parameters"])


        if run_parameters["evaluateonly"]:
            print("Evaluation only mode activated")
            print("Loading trained model")
            LOAD_PATH = os.path.join(submission_dir,"trained_model")
            simulator.restore(path=LOAD_PATH)
            training_time = 0
        else:
            print("Training simulator")
            start = time.time()
            simulator.train(benchmark.train_dataset,
                            benchmark.val_dataset, 
                            save_path=None, 
                            **run_parameters["training_config"]
                            )
            training_time = time.time() - start

            print("Run successfull in " + str(training_time) + " seconds")


        print("Starting evaluation")
        
        start_test = time.time()

        print("Predicting on test dataset")
        predictions = simulator.predict(benchmark._test_dataset)

        env = get_env(get_kwargs_simulator_scenario(benchmark.config))
        evaluator = PowerGridEvaluation(benchmark.config)
        print("Evaluation on test dataset")
        metrics_test = evaluator.evaluate(observations=benchmark._test_dataset.data,
                                        predictions=predictions,
                                        dataset=benchmark._test_dataset,
                                        augmented_simulator=simulator,
                                        env=env)

        print("Metrics ",metrics_test)
        metrics_all = dict()
        metrics_all["test"] = metrics_test


        print("Predicting on test ood dataset")
        predictions_ood = simulator.predict(benchmark._test_ood_topo_dataset)
        evaluator = PowerGridEvaluation(benchmark.config)
        print("Evaluation on test ood dataset")
        metrics_ood = evaluator.evaluate(observations=benchmark._test_ood_topo_dataset.data,
                                        predictions=predictions_ood,
                                        env=env)
        print("Metrics OOD : ", metrics_ood)

        metrics_all["test_ood_topo"] = metrics_ood
    
        # save evaluation for scoring program
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("######################")
        print(metrics_all)
        print("######################")
        json_metrics = json.dumps(eval(str(metrics_all)), indent=4)
        # Writing to sample.json
        with open(os.path.join(output_dir, f'json_metrics_{date_now}.json'), "w") as outfile:
            outfile.write(json_metrics)

        print(json_metrics)
   