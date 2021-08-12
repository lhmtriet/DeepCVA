import re
from glob import glob

import numpy as np
import pandas as pd


def logfile_to_time_df(logfile, groupfunc=None):
    """
    Given logfile, return training times
    average_results: if True, return the average across all partitions/runs

    USAGE:
    logfile = "multitask_sequential_crnn_v3_1.out"
    df = logfile_to_time_df(logfile)
    df.to_csv(logfile + "_time.csv")

    """

    # Final results (to turn into dataframe)
    final_results = []

    # Read logfile
    with open(logfile, "r") as file:
        data_raw = file.read()

    # Split by partitions and runs
    data = data_raw.split("Current partition:")[1:]
    assert len(data) == 10, "Error in [partition] extraction"
    data = [i.split("Run # ")[1:] for i in data]
    assert pd.Series([len(i) == 10 for i in data]).all(), "Error in [run] extraction"

    # Main string parsing loop
    for partition in range(10):
        for run in range(10):

            # Extract train data and evaluate_best_model data
            epoch_data = data[partition][run]
            epoch_data = epoch_data.split("Evaluating the best model")
            assert len(epoch_data) == 3, "Error in [best model evaluation] extraction"
            evaluation_data = epoch_data[2]
            train_data = epoch_data[0]
            train_data = train_data.split("Epoch ")[1:]

            # Test epoch extraction validity
            epoch_nums = [i.split("\n")[0] for i in train_data]
            epoch_n = [int(i.split("/")[0]) for i in epoch_nums]
            epoch_d = [int(i.split("/")[1]) for i in epoch_nums]
            assert all(
                j == i + 1 for i, j in zip(epoch_n, epoch_n[1:])
            ), "Error in [epoch] extraction"
            assert len(set(epoch_d)) == 1, "Error in [epoch] extraction"

            # Extract relevant data
            epoch_times = []
            val_times = []
            for epoch in train_data:

                # Get epoch Time
                epoch_time = [i for i in epoch.split("\n") if "loss: " in i]
                assert len(epoch_time) == 1, "[loss: ] appeared more than once"
                epoch_time = epoch_time[0].split(" - ")[1]
                epoch_time = re.split("(\d+)", epoch_time)[1:]  # Split by digits
                assert (
                    len(epoch_time) == 2 and epoch_time[1] == "s"
                ), "Error in [epochtime]"
                epoch_time = int(epoch_time[0])
                epoch_times.append(epoch_time)

                # Get validation time
                val_time = [i for i in epoch.split("\n") if "Validation time: " in i]
                assert len(val_time) == 1, "Error in [validation time] extraction"
                val_time = re.findall(r"[-+]?\d*\.\d+|\d+", val_time[0])
                assert len(val_time) == 1, "Error in [validation time] extraction"
                val_times.append(float(val_time[0]))

                # Get best models
                if "Saving best model at epoch" in epoch:
                    best_model_at_epoch = int(epoch.split("\n")[0].split("/")[0])

            # Extract final evaluation time
            final_val = [
                i for i in evaluation_data.split("\n") if "Validation time: " in i
            ]
            assert len(final_val) == 1, "Error in [final val time] extraction"
            final_val = re.findall(r"[-+]?\d*\.\d+|\d+", final_val[0])
            assert len(final_val) == 1, "Error in [final val time] extraction"

            # Sanity checks
            assert len(epoch_times) == len(val_times), "Epoch times != validation times"
            assert len(epoch_nums) == len(epoch_times), "Epoch times != num epochs"

            # Calculate all relevant times
            r = {}
            r["partition"] = partition
            r["run"] = run
            r["avg_time_per_epoch"] = np.mean(epoch_times)
            r["number_of_epoch_at_best_model"] = best_model_at_epoch
            r["number_of_epoch_at_finish"] = len(epoch_nums)
            r["total_train_time_at_best_model"] = np.sum(
                epoch_times[:best_model_at_epoch]
            )
            r["total_train_time_at_finish"] = np.sum(epoch_times)
            r["total_val_time_at_best_model"] = np.sum(val_times[:best_model_at_epoch])
            r["total_val_time_at_finish"] = np.sum(val_times)
            r["total_time_at_best_model"] = (
                r["total_train_time_at_best_model"] + r["total_val_time_at_best_model"]
            )
            r["total_time_at_finish"] = (
                r["total_train_time_at_finish"] + r["total_val_time_at_finish"]
            )
            r["final_evaluation_time"] = float(final_val[0])

            final_results.append(r)

    final_results = pd.DataFrame(final_results)
    if groupfunc:
        final_results["temp"] = 1
        if groupfunc == "mean":
            return final_results.groupby("temp").mean().reset_index(drop=True)
        elif groupfunc == "sum":
            return final_results.groupby("temp").sum().reset_index(drop=True)
        else:
            raise Exception("groupby: 'mean' or 'sum'")
    return final_results
