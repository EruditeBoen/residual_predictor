from pathlib import Path
import json
import numpy as np

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
    pass

def build_full_dataset(circuit_ids, hardware_dir, simulated_dir):
    pass

def leave_one_variant_out_eval(df):
    pass

def summarize(fold_results):
    pass

for i in range(1,8):
    average_hardware_runs(f"variant_{i:02d}", "data/hardware", "data/hardware")
