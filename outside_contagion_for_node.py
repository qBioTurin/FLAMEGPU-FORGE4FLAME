import json
import numpy as np
import os
import matplotlib.pyplot as plt

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
    else:
        mean1 = np.random.uniform(10, days/2)
        mean2 = np.random.uniform(mean1 + 10, days-10)
        amplitude1 = np.random.uniform(0, 0.03)
        amplitude2 = np.random.uniform(0, 0.01)
        std1 = np.random.uniform(5, 10)
        std2 = np.random.uniform(5, 10)
        curve = gaussian_curve(x, mean1, amplitude1, std1) + gaussian_curve(x, mean2, amplitude2, std2)
    
    # Clip massimo 0.1 per il picco
    curve = np.clip(curve, 0, 0.05)

    data = [{"day": int(day), "percentage_infected": f"{val:.12f}"} for day, val in zip(x, curve)]
    return data

def generate_files(input_path, output_dir="NODE", n_files=500):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_path, "r") as f:
        base_data = json.load(f)

    for i in range(1, n_files + 1):
        new_curve = generate_random_peak_curve()
        base_data["outside_contagion"] = new_curve
        base_data["starting"][0]["nrun"] = "25"
        dir_path = f"{output_dir}/outside_contagion_{i}"
        os.makedirs(dir_path, exist_ok=True)
        output_path = f"{dir_path}/model.json"
        with open(output_path, "w") as out_f:
            json.dump(base_data, out_f, indent=2)

        days = [d["day"] for d in new_curve]
        values = [float(d["percentage_infected"]) for d in new_curve]

        plt.figure(figsize=(10, 5))
        plt.plot(days, values, label="fraction infected")
        plt.title(f"Outside Contagion {i}")
        plt.xlabel("Day")
        plt.ylabel("Fraction Infected")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        #plt.savefig(f"{output_dir}/outside_contagion_{i}.png")
        plt.close()

# Uso:
generate_files("resources/f4f/Hospital_NoCountermeasures/model.json")
