import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write, read
import matplotlib.pyplot as plt
import os


# PARAMETRELER

fs = 44100
tone_duration = 0.04  
silence_duration = 0.01
threshold = 0.05      


# TÜRKÇE KARAKTER SETİ

characters = [
    "A","B","C","Ç","D","E","F","G","Ğ","H",
    "I","İ","J","K","L","M","N","O","Ö","P",
    "R","S","Ş","T","U","Ü","V","Y","Z"," "
]

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


# ÇEKİRDEK FONKSİYONLAR


def generate_tone(f1, f2):
    t = np.linspace(0, tone_duration, int(fs*tone_duration), endpoint=False)
    return 0.5 * np.sin(2*np.pi*f1*t) + 0.5 * np.sin(2*np.pi*f2*t)

def goertzel(samples, target_freq):
    N = len(samples)
    k = int(0.5 + (N * target_freq) / fs)
    omega = (2.0 * np.pi * k) / N
    coeff = 2.0 * np.cos(omega)
    s_prev, s_prev2 = 0, 0
    for sample in samples:
        s = sample + coeff*s_prev - s_prev2
        s_prev2, s_prev = s_prev, s
    return s_prev2**2 + s_prev**2 - coeff*s_prev*s_prev2


# ENCODER & DECODER


def encode_text(text, filename="encoded.wav"):
    text = text.upper()
    signal = np.array([])
    for char in text:
        if char not in mapping: continue
        f1, f2 = mapping[char]
        tone = generate_tone(f1, f2)
        silence = np.zeros(int(fs*silence_duration))
        signal = np.concatenate((signal, tone, silence))
    
    if len(signal) > 0:
        signal = signal / np.max(np.abs(signal))
        write(filename, fs, (signal * 32767).astype(np.int16))
        print(f"Başarıyla kaydedildi: {filename}")
        sd.play(signal, fs)
        sd.wait()

def decode_audio():
    wav_files = [f for f in os.listdir('.') if f.endswith('.wav')]
    if not wav_files:
        print("\n[HATA]: Klasörde .wav dosyası bulunamadı!")
        return

    print("\n--- Analiz Edilecek Dosyayı Seçin ---")
    for i, f in enumerate(wav_files):
        print(f"{i+1} - {f}")
    
    try:
        secim = int(input("\nDosya numarası: ")) - 1
        filename = wav_files[secim]
    except (ValueError, IndexError):
        print("Geçersiz seçim.")
        return

    fs_read, data = read(filename)
    data = data.astype(np.float32) / 32767.0
    if len(data.shape) > 1: data = data[:, 0]

    window_size = int(fs_read * tone_duration)
    step_size = int(fs_read * (tone_duration + silence_duration))

    decoded_text = ""
    last_char = None 
    
    print(f"\n>>> [{filename}] Çözümleniyor: ", end="", flush=True)

    for start in range(0, len(data) - window_size + 1, step_size):
        segment = data[start:start+window_size]
        if np.max(np.abs(segment)) < threshold: continue

        segment_windowed = segment * np.hamming(len(segment))
        low_powers = [goertzel(segment_windowed, lf) for lf in low_freqs]
        high_powers = [goertzel(segment_windowed, hf) for hf in high_freqs]

        detected_low = low_freqs[np.argmax(low_powers)]
        detected_high = high_freqs[np.argmax(high_powers)]

        char = reverse_mapping.get((detected_low, detected_high), "?")
        
        if char != last_char:
            decoded_text += char
            last_char = char
            print(char, end="", flush=True)
            sd.play(segment, fs_read)
            sd.wait()

    print("\n\n[ANALİZ TAMAMLANDI]")


# VISUALIZATION DASHBOARD


def plot_all_graphs():
    wav_files = [f for f in os.listdir('.') if f.endswith('.wav')]
    if not wav_files:
        print("Dosya bulunamadı.")
        return
    print("\nGrafik Paneli İçin Dosya Seçin:")
    for i, f in enumerate(wav_files):
        print(f"{i+1} - {f}")
    try:
        secim = int(input("Seçim: ")) - 1
        filename = wav_files[secim]
    except: return

    fs_read, data = read(filename)
    data = data.astype(np.float32) / 32767.0
    if len(data.shape) > 1: data = data[:, 0]
    
    time_axis = np.linspace(0, len(data) / fs_read, num=len(data))
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.6)

    # 1. Time Domain
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(time_axis, data, color='#00d1b2', lw=0.7)
    ax1.set_title("DTMF Signal - Time Domain (First 0.5s)", color='white')
    ax1.set_xlim(0, min(0.5, time_axis[-1]))

    # 2. Spectrogram
    ax2 = plt.subplot(3, 1, 2)
    ax2.specgram(data, NFFT=1024, Fs=fs_read, noverlap=512, cmap='magma')
    ax2.set_title("Signal Spectrogram (Heatmap)", color='white')
    ax2.set_ylim(500, 1700)

    # 3. FFT vs Goertzel
    char_samples = int(fs_read * tone_duration)
    first_char = data[:char_samples] * np.hamming(char_samples)
    
    ax3 = plt.subplot(3, 2, 5)
    fft_data = np.abs(np.fft.rfft(first_char))
    fft_freqs = np.fft.rfftfreq(char_samples, 1/fs_read)
    ax3.plot(fft_freqs, fft_data, color='#ffdd57')
    ax3.set_title("FFT Spectrum (1st Char)")
    ax3.set_xlim(500, 1700)

    ax4 = plt.subplot(3, 2, 6)
    all_targets = low_freqs + high_freqs
    g_powers = [goertzel(first_char, f) for f in all_targets]
    g_powers = np.array(g_powers) / max(g_powers) if max(g_powers) > 0 else g_powers
    colors = ['#3273dc']*len(low_freqs) + ['#ff3860']*len(high_freqs)
    ax4.bar([str(f) for f in all_targets], g_powers, color=colors)
    ax4.set_title("Goertzel Filter Output")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ANA MENÜ


def main():
    while True:
        print("\n" + "="*30)
        print(" TURKISH DTMF SYSTEM ")
        print("="*30)
        print("1 - Encode Text")
        print("2 - Decode Audio")
        print("3 - View Signal Dashboard")
        print("0 - Exit")
        choice = input("\nSeçiminiz: ")
        if choice == "1":
            msg = input("Metin: ")
            fname = input("Dosya adı: ")
            encode_text(msg, fname + ".wav")
        elif choice == "2":
            decode_audio()
        elif choice == "3":
            plot_all_graphs()
        elif choice == "0":
            break

if __name__ == "__main__":
    main()
