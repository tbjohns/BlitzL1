import python as blitzl1
import os
import sys
import numpy as np
from scipy import sparse
from sklearn.datasets import load_svmlight_file
import time
import datetime
import blitzl1

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, "../.."))


def format_b(b):
    max_b = max(b)
    min_b = min(b)
    scale = 2.00 / (max_b - min_b)
    return scale * (b - max_b) + 1.0


def get_time(log_dir):
    itr = 0
    while True:
        filepath = "%s/time.%d" % (log_dir, itr)
        try:
            time = float(open(filepath).read())
        except:
            return time
        itr += 1


def save_results(experiment_names, experiment_times, benchmark_name, code_version):
    out_path = "../results/%s_benchmark.%s" % (benchmark_name, code_version)
    out_file = open(out_path, "w")
    timestamp = datetime.datetime.fromtimestamp(
        time.time()).strftime('%Y-%m-%d %H:%M:%S')
    out_file.writelines(timestamp + "\n\n")
    for tag, run_time in zip(experiment_names, experiment_times):
        out_file.writelines("%s: %.5f\n" % (tag, run_time))
    out_file.writelines("\n\n")
    out_file.writelines("Total time: %.5f\n" % sum(experiment_times))
    time_avg = sum(experiment_times)/len(experiment_times)
    out_file.writelines("Average time: %.5f\n" % time_avg)

    times = np.array(experiment_times)
    obj = sum(np.log10(times)) / len(times)
    out_file.writelines("log10-adjusted average: %.5f\n\n" % obj)
    out_file.close()


def main():
    benchmark_name = sys.argv[1]
    code_version = sys.argv[2]
    benchmark_conf_path = "../conf/%s_benchmark" % benchmark_name
    conf_file = open(benchmark_conf_path)
    current_dataset = "none"
    blitzl1.set_verbose(True)
    experiment_names = []
    experiment_times = []
    for line in conf_file:
        print("\n\n", line)
        line_values = line.split()
        dataset_name = line_values[0]
        loss_type = line_values[1]
        lambda_ratio = float(line_values[2])

        if current_dataset != dataset_name:
            data_path = "../data/%s" % dataset_name
            (A, b) = load_svmlight_file(data_path)
            A_csc = sparse.csc_matrix(A)
            current_dataset = dataset_name

        if loss_type == "squared":
            prob = blitzl1.LassoProblem(A_csc, b)
        elif loss_type == "logistic":
            b = format_b(b)
            prob = blitzl1.LogRegProblem(A_csc, b)
        else:
            print("loss function not recognized")
        lammax = prob.compute_lambda_max()

        blitzl1.set_tolerance(1e-5)
        blitzl1.set_use_intercept(True)
        initial_conditions = False
        initial_x = None
        initial_intercept = None
        for option in line_values[3:]:
            (setting, value) = option.split("=")
            if setting == "tolerance":
                blitzl1.set_tolerance(float(value))
            if setting == "intercept":
                value_map = {"false": False, "true": True}
                blitzl1.set_use_intercept(value_map[value])
            if setting == "initial":
                l1_penalty = float(value) * lammax
                sol = prob.solve(l1_penalty)
                initial_x = sol.x
                initial_intercept = sol.intercept

        log_dir = "/tmp/blitzl1_benchmark"
        os.system("rm -rf %s" % log_dir)
        os.mkdir(log_dir)

        l1_penalty = lammax * lambda_ratio
        prob.solve(l1_penalty,
                   log_directory=log_dir,
                   initial_x=initial_x,
                   initial_intercept=initial_intercept)

        experiment_names.append(line.strip())
        time = get_time(log_dir)
        experiment_times.append(time)

    save_results(experiment_names,
                 experiment_times,
                 benchmark_name,
                 code_version)
    conf_file.close()


if __name__ == "__main__":
    main()
