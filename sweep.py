import argparse
from pathlib import Path
import libm2k
import matplotlib.pyplot as plt
import time
import numpy as np
import json

from hardware_setup import make_context, configure_channels

def build_frequency_list(f_start, f_stop, n_points):
    return np.geomspace(f_start, f_stop, num=n_points)

def generate_drive_waveform(frequency_hz, sample_rate, amplitude_v):
    n_cycles = 15
    duration = round(sample_rate/frequency_hz)
    n_samples = n_cycles * duration
    n = np.arange(n_samples)

    wave = amplitude_v * np.sin(2*np.pi*frequency_hz*n/sample_rate)

    return wave

def buffer_size_for_frequency(frequency_hz, sample_rate):
    min_cycles = 15
    samples_per_cycle = round(sample_rate / frequency_hz)
    n_samples = max(min_cycles * samples_per_cycle, 1024)
    n_samples = n_samples + (4 - n_samples % 4) % 4
    return n_samples

def measure_gain_phase(input_samples, output_samples, frequency_hz, sample_rate):
    N = len(input_samples)
    t = np.arange(N) / sample_rate
    ref_cos = np.cos(2*np.pi*frequency_hz*t)
    ref_sin = np.sin(2*np.pi*frequency_hz*t)

    I_input = (2/N) * sum(input_samples * ref_cos)
    Q_input = (2/N) * sum(input_samples * ref_sin)

    amp_input = np.sqrt(I_input**2 + Q_input**2)
    phase_input = np.arctan2(Q_input, I_input)

    N = len(output_samples)
    t = np.arange(N) / sample_rate
    ref_cos = np.cos(2*np.pi*frequency_hz*t)
    ref_sin = np.sin(2*np.pi*frequency_hz*t)

    I_output = (2/N) * sum(output_samples * ref_cos)
    Q_output = (2/N) * sum(output_samples * ref_sin)

    amp_output = np.sqrt(I_output**2 + Q_output**2)
    phase_output = np.arctan2(Q_output, I_output)

    gain = 20*np.log10(amp_output/amp_input)
    phase = (phase_output-phase_input)*(180/np.pi)
    phase = (phase + 180) % 360 - 180

    return {"amp_input": amp_input, "amp_output": amp_output, "phase_input": phase_input, "phase_output": phase_output, "gain_db": gain, "phase_deg": phase}

def synthetic_rlc_response(frequencies, r, l, c):
    res = []
    for f in frequencies:
        w = 2*np.pi*f
        C = w*l - 1/(w*c)
        R = r**2/(r**2 + C**2)
        I = -(r*C / (r**2 + C**2))
        H_f_mag = np.sqrt(R**2 + I**2)
        gain = 20*np.log10(abs(H_f_mag))
        phase = np.arctan2(I, R)*(180/np.pi)
        phase = (phase + 180) % 360 - 180
        res.append({"gain_db": gain, "phase_deg": phase, "frequency_hz": f})

    return res

def sweep(ctx, ain, aout, frequencies):
    ain_sample_rate = 10000000
    aout_sample_rate = 7500000
    res = []
    for hz in frequencies:
        aout.setCyclic(True)
        aout.push(0, generate_drive_waveform(hz, aout_sample_rate, 1))

        settle_cycles = 40
        settle_time = settle_cycles / hz
        time.sleep(max(settle_time, 0.001))

        n_samples = buffer_size_for_frequency(hz, ain_sample_rate)

        ain.startAcquisition(n_samples)
        data = ain.getSamples(n_samples)
        ain.stopAcquisition()

        input_samples = data[0]
        output_samples = data[1]

        measurement = measure_gain_phase(input_samples, output_samples, hz, ain_sample_rate)
        measurement["frequency_hz"] = hz       

        res.append(measurement)

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit-id", type=str)
    parser.add_argument("-r", "--resistance", type=float)
    parser.add_argument("-l", "--inductance", type=float)
    parser.add_argument("-c", "--capacitance", type=float)
    parser.add_argument("-start", "--f-start", type=float)
    parser.add_argument("-stop", "--f-stop", type=float)
    parser.add_argument("-p", "--points", type=int)
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    metadata = {"circuit_id": args.circuit_id, 
                "component_values": {"r_ohms": args.resistance, "l_henries": args.inductance, "c_farads": args.capacitance},
                "sweep_config": {"f_start_hz": args.f_start, "f_stop_hz": args.f_stop, "points": args.points}
    }

    output = args.output
    dry_run = args.dry_run

    f_start = metadata["sweep_config"]["f_start_hz"]
    f_stop = metadata["sweep_config"]["f_stop_hz"]
    n_points = metadata["sweep_config"]["points"]

    freq_lst = build_frequency_list(f_start, f_stop, n_points)

    r = metadata["component_values"]["r_ohms"]
    l = metadata["component_values"]["l_henries"]
    c = metadata["component_values"]["c_farads"]

    if dry_run:
        measurements = synthetic_rlc_response(freq_lst, r, l, c)
    else:
        ctx = make_context()
        try:
            ain, aout = configure_channels(ctx)
            print("AnalogOut max sample rate:", aout.getMaximumSamplerate(0))
            print("AnalogIn max sample rate:", ain.getMaximumSamplerate())
            measurements = sweep(ctx, ain, aout, freq_lst)
        finally:
            libm2k.contextClose(ctx)

    output_dict = {"metadata": metadata, "measurements": measurements}

    out_path = Path(output)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(output_dict, f, indent=2)

if __name__ == "__main__":
    main()
