import threading
import time
import sys
import queue
import readchar
import tempfile, os
import numpy as np
import sounddevice as sd
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import lameenc

running = True
paused = False
vol = 50
SAMPLE_RATE = 44100
plot_queue = queue.Queue()
played_once = False

def countdown(duration):
    total = int(duration)
    for i in range(5, 0, -1):
        mins, secs = divmod(i, 60)
        bar_filled = '#' * i
        bar_empty  = '.' * (5 - i)
        remaining_mins, remaining_secs = divmod(total, 60)
        sys.stdout.write(f"\r  RECORDING IN  [{bar_filled}{bar_empty}]  {mins:02d}:{secs:02d}  |  REMAINING  {remaining_mins:02d}:{remaining_secs:02d}  ")
        sys.stdout.flush()
        time.sleep(1)
        total -= 1
    sys.stdout.write(f"\r  [#####]  00:00  |  REMAINING  {total//60:02d}:{total%60:02d}  GO!      \n")
    sys.stdout.flush()

def save_recording_wav(audio, path=None):
    if path is None:
        path = tempfile.mktemp(suffix=".wav")
    sf.write(path, audio, SAMPLE_RATE)
    print(f"\n💾 WAV saved → {path}")
    return path

def save_recording_mp3(audio):
    folder = os.path.expanduser("~/Downloads/SineSweep")
    os.makedirs(folder, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"recording_{timestamp}.mp3")
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(128)
    encoder.set_in_sample_rate(SAMPLE_RATE)
    encoder.set_channels(1)
    encoder.set_quality(2)
    mp3_data = encoder.encode(audio.tobytes())
    mp3_data += encoder.flush()
    with open(path, 'wb') as f:
        f.write(mp3_data)
    print(f"🎵 MP3 saved → {path}")

def recording_to_df(wav_path):
    samples, sr = sf.read(wav_path, dtype='int16')
    df = pd.DataFrame(samples, columns=['mono'])
    df['time_s'] = df.index / sr
    print(f"📊 {len(df)} samples, {df['time_s'].iloc[-1]:.2f}s")
    return df

def compute_fft_bars(samples, n_bars=60):
    fft_vals  = np.abs(np.fft.rfft(samples.flatten()))
    fft_freqs = np.fft.rfftfreq(len(samples.flatten()), d=1/SAMPLE_RATE)
    mask      = (fft_freqs >= 20) & (fft_freqs <= 20000)
    fft_vals  = fft_vals[mask]
    fft_freqs = fft_freqs[mask]
    log_edges = np.logspace(np.log10(20), np.log10(20000), n_bars + 1)
    bar_freqs, bar_vals = [], []
    for i in range(n_bars):
        bin_mask = (fft_freqs >= log_edges[i]) & (fft_freqs < log_edges[i+1])
        bar_freqs.append((log_edges[i] + log_edges[i+1]) / 2)
        bar_vals.append(fft_vals[bin_mask].mean() if bin_mask.any() else 0)
    bar_vals = np.array(bar_vals)
    bar_vals = bar_vals / bar_vals.max() if bar_vals.max() > 0 else bar_vals
    return bar_freqs, bar_vals, log_edges

def print_ascii_chart(title, bar_vals, n_bars=60, bar_height=12):
    label_freqs = [20, 100, 500, 1000, 5000, 10000, 20000]
    print(f"\n  {title}\n")
    for row in range(bar_height, 0, -1):
        line = "  "
        for val in bar_vals:
            line += "#" if val * bar_height >= row else " "
        print(line)
    print("  " + "-" * n_bars)
    label_row = "  "
    last_pos  = 0
    for lf in label_freqs:
        pos = int((np.log10(lf) - np.log10(20)) / (np.log10(20000) - np.log10(20)) * (n_bars - 1))
        gap = pos - last_pos
        label = f"{lf}Hz" if lf < 1000 else f"{lf//1000}kHz"
        label_row += " " * max(0, gap - len(label) + 1) + label
        last_pos = pos + 1
    print(label_row)
    print()

def _process_recording(audio, playback):
    wav_path = save_recording_wav(audio)
    save_recording_mp3(audio)
    df = recording_to_df(wav_path)

    bar_freqs, bar_vals, _ = compute_fft_bars(df['mono'].values)
    _, pb_bar_vals, _      = compute_fft_bars(playback)

    print_ascii_chart("MICROPHONE  20Hz → 20kHz", bar_vals)
    print_ascii_chart("PLAYBACK    20Hz → 20kHz", pb_bar_vals)

    print("  Press [p] to play again  |  any other key to continue")
    print()

    plot_queue.put((bar_freqs, bar_vals, pb_bar_vals))

def plot_if_ready():
    try:
        bar_freqs, bar_vals, pb_bar_vals = plot_queue.get_nowait()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        for ax, vals, title, color in [
            (ax1, bar_vals,    'Microphone  20Hz → 20kHz', 'lime'),
            (ax2, pb_bar_vals, 'Playback    20Hz → 20kHz', 'cyan'),
        ]:
            ax.bar(range(len(vals)), vals, width=0.9, color=color, edgecolor='none')
            ax.set_facecolor('black')
            ax.set_title(title, color=color, fontfamily='monospace', fontsize=13)
            ax.set_ylabel('Amplitude (normalised)', color=color, fontfamily='monospace')
            ax.tick_params(colors=color)
            ax.set_ylim(0, 1)
            n_bars = len(vals)
            log_edges = np.logspace(np.log10(20), np.log10(20000), n_bars + 1)
            bar_freqs_arr = (log_edges[:-1] + log_edges[1:]) / 2
            label_freqs = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
            tick_pos, tick_labels = [], []
            for lf in label_freqs:
                idx = np.argmin(np.abs(bar_freqs_arr - lf))
                tick_pos.append(idx)
                tick_labels.append(f"{lf}Hz" if lf < 1000 else f"{lf//1000}kHz")
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, fontfamily='monospace', color=color, fontsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)

        fig.patch.set_facecolor('black')
        plt.tight_layout()
        plt.show()
    except queue.Empty:
        pass

def play_sound():
    global played_once
    tmp = "sine.wav"
    audio, sr = sf.read(tmp, dtype='int16')
    audio = audio.flatten()
    duration = len(audio) / sr
    countdown(duration)

    recorded     = np.zeros((len(audio), 1), dtype='int16')
    played_audio = np.zeros(len(audio), dtype='int16')
    idx = [0]

    def callback(indata, outdata, frames, time_info, status):
        end   = idx[0] + frames
        chunk = audio[idx[0]:end]
        scaled = (chunk * (vol / 100)).astype('int16')
        if len(scaled) < frames:
            outdata[:len(scaled)] = scaled.reshape(-1, 1)
            outdata[len(scaled):] = 0
            recorded[idx[0]:idx[0]+len(scaled)]     = indata[:len(scaled)]
            played_audio[idx[0]:idx[0]+len(scaled)] = scaled
        else:
            outdata[:] = scaled.reshape(-1, 1)
            recorded[idx[0]:end]     = indata
            played_audio[idx[0]:end] = scaled
        idx[0] = end

    with sd.Stream(samplerate=sr, channels=1, dtype='int16', callback=callback):
        sd.sleep(int(duration * 1000))

    rec_thread = threading.Thread(target=_process_recording, args=(recorded, played_audio), daemon=True)
    rec_thread.start()
    rec_thread.join()
    played_once = True

def render_vol():
    filled = round(vol / 100 * 30)
    bar = '[' + '#' * filled + '-' * (30 - filled) + ']'
    sys.stdout.write(f"\r  VOL {bar} {vol:3d}%  ")
    sys.stdout.flush()

def my_program():
    global played_once
    if not played_once:
        play_sound()
    else:
        time.sleep(0.5)

def worker():
    while running:
        if not paused:
            my_program()

def control_loop():
    global running, paused, vol
    print("Controls: [SPACE]=pause/resume  [p]=play again  [q]=quit  [+/-]=volume")
    while running:
        plot_if_ready()
        render_vol()
        key = readchar.readkey()
        if key == ' ':
            paused = not paused
            print("\n⏸ Paused" if paused else "\n▶ Resumed")
        elif key in ('p', 'P'):
            print("\n▶ Replaying...")
            threading.Thread(target=play_sound, daemon=True).start()
        elif key == 'q':
            running = False
            print("\n⏹ Quit")
        elif key in ('+', readchar.key.RIGHT):
            vol = min(100, vol + 2)
        elif key in ('-', readchar.key.LEFT):
            vol = max(0, vol - 2)

thread = threading.Thread(target=worker, daemon=True)
thread.start()
control_loop()