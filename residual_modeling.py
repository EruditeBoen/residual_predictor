from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


def average_hardware_runs(circuit_id, hardware_dir, output_dir):
    hw_paths = [Path(hardware_dir) / f"{circuit_id}_run{i}.json" for i in range(1, 4)]

    metadata = []

    hw_freqs = []
    hw_gains = []
    hw_phases = []

    for i, hw_path in enumerate(hw_paths):
        with open(hw_path) as f:
            hw_data = json.load(f)
            if i == 0:
                metadata = hw_data["metadata"]
                hw_freqs = [m["frequency_hz"] for m in hw_data["measurements"]]
            hw_gain = [m["gain_db"] for m in hw_data["measurements"]]
            hw_phase = [m["phase_deg"] for m in hw_data["measurements"]]
            hw_gains.append(hw_gain)
            hw_phases.append(hw_phase)

    avg_hw_gains = np.mean(hw_gains, axis=0)
    avg_hw_phases = np.mean(hw_phases, axis=0)

    measurements = []

    for i in range(len(hw_freqs)):
        measurements.append({"gain_db": float(avg_hw_gains[i]), "phase_deg": float(avg_hw_phases[i]), "frequency_hz": hw_freqs[i]})

    output_dict = {"metadata": metadata, "measurements": measurements}

    out_path = Path(output_dir) / f"{circuit_id}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(output_dict, f, indent=2)


def build_variant_table(circuit_id, hardware_dir, simulated_dir):
    hw_path = Path(hardware_dir) / f"{circuit_id}.json" 
    sim_path = Path(simulated_dir) / f"{circuit_id}_sim.json"

    with open(hw_path) as f:
        hw_data = json.load(f)

    with open(sim_path) as f:
        sim_data = json.load(f)

    metadata = hw_data["metadata"]
    measurements = hw_data["measurements"]

    simulated = sim_data["measurements"]

    r, l, c = metadata["component_values"]["r_ohms"], metadata["component_values"]["l_henries"], metadata["component_values"]["c_farads"]

    frequency_hz = np.array([m["frequency_hz"] for m in measurements])
    gain_db = np.array([m["gain_db"] for m in measurements])
    phase_deg = np.array([m["phase_deg"] for m in measurements])

    sim_freqs = np.array([m["frequency_hz"] for m in simulated])
    sim_gain = np.array([m["gain_db"] for m in simulated])
    sim_phase = np.array([m["phase_deg"] for m in simulated])

    sorted_sim_freqs_idx = np.argsort(sim_freqs)

    sorted_sim_freqs = sim_freqs[sorted_sim_freqs_idx]
    sorted_sim_gain = sim_gain[sorted_sim_freqs_idx]
    sorted_sim_phase = sim_phase[sorted_sim_freqs_idx]

    log_sim_freqs = np.log10(sorted_sim_freqs)

    output = []

    for f, g, p in zip(frequency_hz, gain_db, phase_deg):
        log_hw_freq = np.log10(f)

        sim_g_interp = np.interp(log_hw_freq, log_sim_freqs, sorted_sim_gain)
        sim_p_interp = np.interp(log_hw_freq, log_sim_freqs, sorted_sim_phase)

        residual_gain = g - sim_g_interp

        output.append({
            "circuit_id": circuit_id,
            "r_ohms": r, "l_henries": l, "c_farads": c,
            "frequency_hz": f, "log_frequency_hz": log_hw_freq,
            "sim_gain_db": sim_g_interp, "sim_phase_deg": sim_p_interp,
            "measured_gain_db": g, "residual_gain_db": residual_gain,
            })

    return output

def build_full_dataset(circuit_ids, hardware_dir, simulated_dir):
    rows = []
    for variant in circuit_ids:
        variant_table = build_variant_table(variant, hardware_dir, simulated_dir)
        rows.extend(variant_table)
    df = pd.DataFrame(rows)
    return df

def leave_one_variant_out_eval(df):
    FEATURE_COLUMNS = ["r_ohms", "l_henries", "c_farads", "log_frequency_hz", "sim_gain_db", "sim_phase_deg"]
    TARGET_COLUMN = "residual_gain_db"

    variants = df["circuit_id"].unique()
    fold_results = []

    for held_out_variant in variants:
        train_df = df[df["circuit_id"] != held_out_variant]
        test_df = df[df["circuit_id"] == held_out_variant]

        train_X = train_df[FEATURE_COLUMNS]
        train_y = train_df[TARGET_COLUMN]
        test_X = test_df[FEATURE_COLUMNS]
        test_y = test_df[TARGET_COLUMN]

        model = GradientBoostingRegressor(
                n_estimators=25,
                learning_rate=0.01,
                max_depth=2,
                random_state=42
        )
        model.fit(train_X, train_y)

        prediction = model.predict(test_X)

        model_mae = mean_absolute_error(test_y, prediction)
        naive_baseline_mae = mean_absolute_error(test_y, np.zeros_like(test_y))

        fold_results.append({"held_out_variant": held_out_variant, "model_mae": model_mae, "naive_baseline_mae": naive_baseline_mae})

    return fold_results

def summarize(fold_results):
    model_maes = [r["model_mae"] for r in fold_results]
    baseline_maes = [r["naive_baseline_mae"] for r in fold_results]

    overall_model_mae = np.mean(model_maes)
    overall_baseline_mae = np.mean(baseline_maes)

    for r in fold_results:
        print(r)

    print(f"Overall model MAE: {overall_model_mae:.4f}")
    print(f"Overall baseline MAE: {overall_baseline_mae:.4f}")
    print(f"Model beats baseline: {overall_model_mae < overall_baseline_mae}")

    return overall_model_mae < overall_baseline_mae

df = build_full_dataset(["variant_01", "variant_02", "variant_03", "variant_04", "variant_05", "variant_06", "variant_07"], "data/hardware", "data/simulated")

results = leave_one_variant_out_eval(df)

summarize(results)
