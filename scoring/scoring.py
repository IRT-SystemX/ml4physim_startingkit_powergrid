#!/usr/bin/env python
# Use directly the outputfile from the LIPS evaluation program.
# Some libraries and options
import os
from sys import argv
import argparse
import sys
import json
import math
import numpy as np

import utils.compute_score as cs

# Default I/O directories:
root_dir = os.path.abspath(os.path.curdir)
default_submission_dir = root_dir + "/submission_results"
default_output_dir = root_dir + "/scoring_output"

def _HERE(*args):
    h = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(h, *args)

class ModelApiError(Exception):
    """Model api error"""

    def __init__(self, msg=""):
        self.msg = msg
        print(msg)


def import_metrics(json_metrics):
    # path_submission_parameters = os.path.join(input_dir, 'json_metrics.json')
    print(json_metrics)
    if not os.path.exists(json_metrics):
        raise ModelApiError("Missing json_metrics.json file")
        exit_program()
    with open(json_metrics) as json_file:
        metrics = json.load(json_file)
    return metrics

def create_parser():
    this_parser = argparse.ArgumentParser()
    
    this_parser.add_argument("--submission_dir", type=str, default=default_submission_dir,
                             help="the directory containing a valid submission from competition")
    this_parser.add_argument("--submission_name", type=str,
                             help="the name of submission that should be executed here")
    this_parser.add_argument("--output_dir", type=str, default=default_output_dir,
                             help="output directory where the results of evaluation should be saved")
    
    return this_parser


# =============================== MAIN ========================================

if __name__ == "__main__":

    #### INPUT/OUTPUT: Get input and output directory names
    parser = create_parser()
    args = parser.parse_args()
    command = ' '.join([sys.executable] + sys.argv)
    
    submission_dir = str(args.submission_dir)
    submission_name = str(args.submission_name)
    output_dir = str(args.output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_runs = os.listdir(os.path.join(submission_dir, submission_name))
    global_scores_all = []
    for run_ in all_runs: 
        score_file = open(os.path.join(output_dir, f'scores_{submission_name}.txt'), 'w')
        print("Scoring file : ", os.path.join(output_dir, f'scores_{submission_name}.txt'))
        metrics = import_metrics(os.path.join(submission_dir, submission_name, run_))
        metrics["solver_time"] = 32.79

        globalScore, test_ml_subscore, test_physics_subscore, test_ood_ml_subscore, test_ood_physics_subscore, speedup_score, speed_up, test_results_disc, test_ood_results_disc = cs.compute_global_score(metrics)
        global_scores_all.append(globalScore)
        print("scoring done ", globalScore, test_ml_subscore, test_physics_subscore, test_ood_ml_subscore, test_ood_physics_subscore, speedup_score)

        score_file.write("global" + ": %0.12f\n" % globalScore)
        score_file.write("ML_test" + ": %0.12f\n" % test_ml_subscore)
        score_file.write("Physics_test" + ": %0.12f\n" % test_physics_subscore)
        score_file.write("ML_ood" + ": %0.12f\n" % test_ood_ml_subscore)
        score_file.write("Physics_ood" + ": %0.12f\n" % test_ood_physics_subscore)
        score_file.write("speedup" + ": %0.12f\n" % speedup_score)
        score_file.write(f"SpeedUp Ratio:  {speed_up:.12f}\n")
        score_file.write("Test_results_disc :" + str(test_results_disc) + "\n")
        score_file.write("Test OOD results disc" + str(test_ood_results_disc) + "\n")
        
    mean_global_scores = np.mean(global_scores_all)
    std_glboal_scores = np.std(global_scores_all)
    score_file.write(f"Mean Global score : {mean_global_scores:.12f}\n")
    score_file.write(f"Std Global score : {std_glboal_scores:.12f}\n")
    score_file.close()
    