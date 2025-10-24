import json
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import csv

def gaussian_curve(x, mean, amplitude, std):
    return amplitude * np.exp(-0.5 * ((x - mean) / std)**2)

def generate_random_peak_curve(days=105):
    x = np.arange(1, days+1)
    single_peak = np.random.choice([True, False])
    if single_peak:
        mean = np.random.uniform(10, days-10)
        amplitude = np.random.uniform(0, 0.03)
        std = np.random.uniform(5, 10)
        curve = gaussian_curve(x, mean, amplitude, std)
        # Pad second peak parameters with zeros
        curve_params = {
            "mean1": mean, "amplitude1": amplitude, "std1": std,
            "mean2": 0, "amplitude2": 0, "std2": 0
        }
    else:
        mean1 = np.random.uniform(10, days/2)
        mean2 = np.random.uniform(mean1 + 10, days-10)
        amplitude1 = np.random.uniform(0, 0.03)
        amplitude2 = np.random.uniform(0, 0.01)
        std1 = np.random.uniform(5, 10)
        std2 = np.random.uniform(5, 10)
        curve = gaussian_curve(x, mean1, amplitude1, std1) + gaussian_curve(x, mean2, amplitude2, std2)
        # Save params for both peaks
        curve_params = {
            "mean1": mean1, "amplitude1": amplitude1, "std1": std1,
            "mean2": mean2, "amplitude2": amplitude2, "std2": std2
        }
    curve = np.clip(curve, 0, 0.05)
    data = [{"day": int(day), "percentage_infected": f"{val:.12f}"} for day, val in zip(x, curve)]
    return data, curve_params

def generate_files(input_path, output_dir="NODE", n_files=200, params_csv="contagion_params.csv"):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_path, "r") as f:
        base_data = json.load(f)

    params_list = []

    for i in range(1, n_files + 1):
        new_curve, curve_params = generate_random_peak_curve()
        base_data["outside_contagion"] = new_curve
        base_data["starting"][0]["nrun"] = "10"
        dir_path = f"{output_dir}/outside_contagion_{i}"
        os.makedirs(dir_path, exist_ok=True)
        output_path = f"{dir_path}/model.json"
        with open(output_path, "w") as out_f:
            json.dump(base_data, out_f, indent=2)

        days = [d["day"] for d in new_curve]
        values = [float(d["percentage_infected"]) for d in new_curve]

        # plt.figure(figsize=(10, 5))
        # plt.plot(days, values, label="fraction infected")
        # plt.title(f"Outside Contagion {i}")
        # plt.xlabel("Day")
        # plt.ylabel("Fraction Infected")
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(f"{output_dir}/outside_contagion_{i}.png")
        # plt.close()

        # Add curve parameters and index
        curve_params["index"] = i
        params_list.append(curve_params)

    # Save all parameters to a CSV
    with open("NODE/" + params_csv, "w", newline="") as csvfile:
        fieldnames = ["index", "mean1", "amplitude1", "std1", "mean2", "amplitude2", "std2"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in params_list:
            writer.writerow(row)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
generate_files("resources/f4f/Hospital_NoCountermeasures/model.json")
