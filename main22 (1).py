import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write, read
import matplotlib.pyplot as plt


# PARAMETRELER
fs = 44100
tone_duration = 0.04  # 40 ms — ödev sınırı içinde
silence_duration = 0.01
threshold = 0.05      # Bir segmenti ton olarak kabul etmek için gereken minimum enerji

# TURKISH CHARACTER SET

characters = [
    "A","B","C","Ç","D","E","F","G","Ğ","H",
    "I","İ","J","K","L","M","N","O","Ö","P",
    "R","S","Ş","T","U","Ü","V","Y","Z"," "
]

# 6 Low Freqs x 5 High Freqs = 30 Characters
low_freqs  = [600, 700, 800, 900, 1000, 1100]
high_freqs = [1200, 1300, 1400, 1500, 1600]

mapping = {}
reverse_mapping = {}

index = 0
for lf in low_freqs:
    for hf in high_freqs:
        if index < len(characters):
            char = characters[index]
            mapping[char] = (lf, hf)
            reverse_mapping[(lf, hf)] = char
            index += 1


# DSP CORE FUNCTIONS

def generate_tone(f1, f2):
    t = np.linspace(0, tone_duration, int(fs*tone_duration), endpoint=False)
    # Balanced dual-sine wave
    return 0.5 * np.sin(2*np.pi*f1*t) + 0.5 * np.sin(2*np.pi*f2*t)

def goertzel(samples, target_freq):
    """The Goertzel algorithm: efficient way to detect a specific frequency."""
    N = len(samples)
    k = int(0.5 + (N * target_freq) / fs)
    omega = (2.0 * np.pi * k) / N
    coeff = 2.0 * np.cos(omega)

    s_prev = 0
    s_prev2 = 0

    for sample in samples:
        s = sample + coeff*s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s

    power = s_prev2**2 + s_prev**2 - coeff*s_prev*s_prev2
    return power


# ENCODER & DECODER

def encode_text(text, filename="encoded.wav"):
    text = text.upper()
    signal = np.array([])

    for char in text:
        if char not in mapping:
            print(f"Skipping unsupported character: {char}")
            continue

        f1, f2 = mapping[char]
        tone = generate_tone(f1, f2)
        silence = np.zeros(int(fs*silence_duration))
        signal = np.concatenate((signal, tone, silence))

    # Normalize to prevent clipping
    if len(signal) > 0:
        signal = signal / np.max(np.abs(signal))
        write(filename, fs, (signal * 32767).astype(np.int16)) # Save as 16-bit PCM
        print(f"Successfully encoded '{text}' to {filename}")
        sd.play(signal, fs)
        sd.wait()

def decode_audio(filename="encoded.wav"):
    fs_read, data = read(filename)
    # Convert back to float -1 to 1
    data = data.astype(np.float32) / 32767.0
    if len(data.shape) > 1: data = data[:, 0]

    window_size = int(fs_read * tone_duration)
    step_size = int(fs_read * (tone_duration + silence_duration))

    decoded_text = ""
    last_char = None 
    
    for start in range(0, len(data) - window_size + 1, step_size):
        segment = data[start:start+window_size]
        
        # Energy Check
        if np.max(np.abs(segment)) < threshold:
            continue

        # Apply Hamming window to reduce spectral leakage
        segment = segment * np.hamming(len(segment))

        low_powers = [goertzel(segment, lf) for lf in low_freqs]
        high_powers = [goertzel(segment, hf) for hf in high_freqs]

        detected_low = low_freqs[np.argmax(low_powers)]
        detected_high = high_freqs[np.argmax(high_powers)]

        char = reverse_mapping.get((detected_low, detected_high), "?")
        if char != last_char:
         decoded_text += char
         last_char = char

    print(f"\n[DECODED]: {decoded_text}")
    return decoded_text


# VISUALIZATION DASHBOARD


def plot_all_graphs(filename="encoded.wav"):
    try:
        fs_read, data = read(filename)
    except FileNotFoundError:
        print("Error: Encoded file not found. Please encode text first.")
        return

    data = data.astype(np.float32) / 32767.0
    if len(data.shape) > 1: data = data[:, 0]
    
    time_axis = np.linspace(0, len(data) / fs_read, num=len(data))

    plt.style.use('dark_background') # Professional aesthetic
    fig = plt.figure(figsize=(14, 10))
    plt.subplots_adjust(hspace=0.5)

    # 1. TIME DOMAIN (First 0.5s)
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(time_axis, data, color='#00d1b2', lw=0.7)
    ax1.set_title("DTMF Signal - Time Domain (First 0.5s)", fontsize=14, color='white')
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, min(0.5, time_axis[-1]))
    ax1.grid(alpha=0.2)

    # 2. SPECTROGRAM (Frequency over Time)
    ax2 = plt.subplot(3, 1, 2)
    Pxx, freqs_spec, bins, im = ax2.specgram(data, NFFT=1024, Fs=fs_read, noverlap=512, cmap='magma')
    ax2.set_title("Signal Spectrogram (Heatmap)", fontsize=14, color='white')
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_ylim(500, 1700) # Zoom into our target range
    plt.colorbar(im, ax=ax2, label='Intensity (dB)')

    # 3. FFT (Left Bottom)
    char_samples = int(fs_read * tone_duration)
    first_char = data[:char_samples] * np.hamming(char_samples)
    fft_data = np.abs(np.fft.rfft(first_char))
    fft_freqs = np.fft.rfftfreq(char_samples, 1/fs_read)

    ax3 = plt.subplot(3, 2, 5)
    ax3.plot(fft_freqs, fft_data, color='#ffdd57')
    ax3.set_title("FFT Spectrum (1st Character)", fontsize=12)
    ax3.set_xlim(500, 1700)
    ax3.set_ylabel("Magnitude")
    for f in low_freqs + high_freqs:
        ax3.axvline(x=f, color='white', linestyle='--', alpha=0.1)

    # 4. GOERTZEL BARS (Right Bottom)
    all_targets = low_freqs + high_freqs
    g_powers = [goertzel(first_char, f) for f in all_targets]
    g_powers = np.array(g_powers) / max(g_powers) if max(g_powers) > 0 else g_powers

    ax4 = plt.subplot(3, 2, 6)
    colors = ['#3273dc']*len(low_freqs) + ['#ff3860']*len(high_freqs)
    ax4.bar([str(f) for f in all_targets], g_powers, color=colors)
    ax4.set_title("Goertzel Filter Power Output", fontsize=12)
    ax4.set_ylabel("Normalized Energy")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# MAIN MENU


def main():
    while True:
        print("\n" + "="*30)
        print(" TURKISH DTMF SYSTEM ")
        print("="*30)
        print("1 - Encode Text")
        print("2 - Decode Audio")
        print("3 - View Signal Dashboard")
        print("0 - Exit")
        
        choice = input("\nSelect Option: ")

        if choice == "1":
            msg = input("Enter Text (Turkish supported): ")
            fname = input("Enter filename (e.g. mesaj1): ")
            encode_text(msg, fname + ".wav")
        elif choice == "2":
            decode_audio()
        elif choice == "3":
            plot_all_graphs()
        elif choice == "0":
            break
        else:
            print("Invalid Option.")

if __name__ == "__main__":
    main()
