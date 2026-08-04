import argparse
import libm2k
import matplotlib.pyplot as plt
import time
import numpy as np
    
def make_context():
    ctx = libm2k.m2kOpen()
    if ctx is None:
        raise RuntimeError("No ADALM2000 device available.")
    
    ctx.reset()
    ctx.resetCalibration()
    
    if ctx.hasContextCalibration():
        ctx.calibrateFromContext()
    else:
        try:
            ctx.calibrateADC()
            ctx.calibrateDAC()
        except Exception as e:
            libm2k.contextCloseAll()
            print("Closing Context")
            return None
        
    print("ADALM2000 has been reset and calibrated.")
    return ctx

def configure_channels(ctx):
    ain = ctx.getAnalogIn()
    aout = ctx.getAnalogOut()
    
    ain.enableChannel(0, True)
    ain.enableChannel(1, True)
    ain.setSampleRate(10000000)
    ain.setRange(0, -3, 3)
    ain.setRange(1, -3, 3)
    
    aout.setSampleRate(0, 7500000)
    aout.enableChannel(0, True)

    print("Requsted/actual AnalogIn rate:", ain.getSampleRate())
    print("Requsted/actual AnalogOut rate:", aout.getSampleRate(0))

    return (ain, aout)

def generate_drive_waveform(frequency_hz, sample_rate, amplitude_v, n_samples=None):
    if n_samples is not None:
        n = np.arange(n_samples)
    else:
        n_cycles = 15
        duration = round(sample_rate/frequency_hz)
        n_samples = n_cycles * duration
        n = np.arange(n_samples)

    wave = amplitude_v * np.sin(2*np.pi*frequency_hz*n/sample_rate)

    return wave

def measure_frequency_via_zero_crossings(samples, sample_rate):
    zero_counts = 0
    amplitude_estimate = (max(samples) - min(samples)) / 2
    hysteresis = 0.075 * amplitude_estimate
    below = samples[0] < -hysteresis
    times = []
    for i in range(1, len(samples)):
        if below and samples[i] > hysteresis:
            zero_counts += 1
            time = ((i/sample_rate)+(i-1)/sample_rate)/2
            times.append(time)
            below = False
        elif samples[i] < -hysteresis:
            below = True

    n_periods = zero_counts-1
    frequency = n_periods/(times[-1]-times[0])
    
    return frequency

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--frequency", type=int, default=2500)
    parser.add_argument("-c", "--calibrate", action="store_true")
    args = parser.parse_args()

    frequency_hz = args.frequency

    if args.calibrate:
        ctx = make_context()
    else:
        ctx = libm2k.m2kOpen()
    ain, aout = configure_channels(ctx)
    buffer = generate_drive_waveform(frequency_hz, 7500000, 1)

    aout.setCyclic(True)
    aout.push(0, buffer.tolist())

    time.sleep(0.5)

    ain.startAcquisition(1000)
    data = ain.getSamples(1000)
    ain.stopAcquisition()

    # for i in range(10):
    #     data = ain.getSamples(1000)
    #     plt.plot(data[0])
    #     plt.plot(data[1])
    #     plt.show()
    #     time.sleep(0.1)

    freq_estimate = measure_frequency_via_zero_crossings(data[0], 7500000)

    print(f"The measured frequency has a {(abs(freq_estimate-frequency_hz)/frequency_hz)*100}% error from the actual frequency.")

    libm2k.contextClose(ctx)

if __name__ == "__main__":
    main()
