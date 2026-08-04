import json
import argparse
import numpy as np
from pathlib import Path
import subprocess
import tempfile

def build_netlist(circuit_id, r, l, c, f_start, f_stop, points, raw_output_path):
    decades = np.log10(f_stop / f_start)
    points_per_decade = max(1, round(points / decades))
    
    netlist = f"""RLC bandpass sweep - {circuit_id}
V1 in 0 AC 1
L1 in mid {l}
C1 mid out {c}
R1 out 0 {r}
.ac dec {points_per_decade} {f_start} {f_stop}
.control
run
wrdata {raw_output_path} vdb(out) vp(out)
.endc
.end
"""

    return netlist


def run_ngspice(netlist_text, work_dir):
    out_path = Path(work_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    netlist_path = out_path / "circuit.cir"
    netlist_path.write_text(netlist_text)

    result = subprocess.run(["ngspice_con", "-b", str(netlist_path)], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ngspice failed:\n{result.stderr}")

def parse_wrdata_output(raw_output_path):
    data = np.loadtxt(raw_output_path)
    freq = data[:, 0]
    vdb_out = data[:, 1]
    vp_out = data[:, 3]*(180/np.pi)
    return (freq, vdb_out, vp_out)

def simulate(circuit_id, circuit_type, r, l, c, f_start, f_stop, points):
    metadata = {"circuit_id": circuit_id, 
                "component_values": {"r_ohms": r, "l_henries": l, "c_farads": c}, 
                "sweep_config": {"f_start_hz": f_start, "f_stop_hz": f_stop, "points": points}}

    measurements = []
    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "sweep_raw.txt"
        netlist_text = build_netlist(circuit_id, r, l, c, f_start, f_stop, points, output_path)
        run_ngspice(netlist_text, tmp)
        frequency_array, gaindb_array, phasedeg_array = parse_wrdata_output(output_path)

        for f, g, p in zip(frequency_array, gaindb_array, phasedeg_array):
            measurements.append({"frequency_hz": float(f), "gain_db": float(g), "phase_deg": float(p)})

    output_dict = {"metadata": metadata, "measurements": measurements}


    return output_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit-id", type=str)
    parser.add_argument("-r", "--resistance", type=float)
    parser.add_argument("--circuit-type", type=str)
    parser.add_argument("-l", "--inductance", type=float)
    parser.add_argument("-c", "--capacitance", type=float)
    parser.add_argument("-start", "--f-start", type=float)
    parser.add_argument("-stop", "--f-stop", type=float)
    parser.add_argument("-p", "--points", type=int)
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()

    circuit_id = args.circuit_id
    circuit_type = args.circuit_type
    r = args.resistance
    l = args.inductance
    c = args.capacitance
    f_start = args.f_start
    f_stop = args.f_stop
    points = args.points
    output = args.output

    output_dict = simulate(circuit_id, circuit_type, r, l, c, f_start, f_stop, points)

    out_path = Path(output)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(output_dict, f, indent=2)


if __name__ == "__main__":
    main()
