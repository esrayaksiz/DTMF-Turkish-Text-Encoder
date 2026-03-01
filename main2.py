import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write, read
import time  # Harfler arasında duraksama yapmak için eklendi

# =========================
# PARAMETERS
# =========================
fs = 44100                
tone_duration = 0.04      
silence_duration = 0.01   
threshold = 0.02          

# =========================
# TURKISH CHARACTER SET
# =========================
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
        char = characters[index]
        mapping[char] = (lf, hf)
        reverse_mapping[(lf, hf)] = char
        index += 1

# =========================
# ENCODER
# =========================
def generate_tone(f1, f2):
    t = np.linspace(0, tone_duration, int(fs*tone_duration), endpoint=False)
    tone = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)
    return tone

def encode_text(text, filename):
    signal = np.array([])
    for char in text.upper():
        if char not in mapping: continue
        f1, f2 = mapping[char]
        tone = generate_tone(f1, f2)
        silence = np.zeros(int(fs*silence_duration))
        signal = np.concatenate((signal, tone, silence))

    signal = signal / np.max(np.abs(signal))
    if not filename.endswith(".wav"): filename += ".wav"
    write(filename, fs, signal.astype(np.float32))
    print(f"\n✅ '{filename}' kaydedildi. Dinlemek için menüden 2'yi seçin.")

# =========================
# GOERTZEL ALGORITHM
# =========================
def goertzel(samples, target_freq, fs):
    N = len(samples)
    k = int(0.5 + (N * target_freq) / fs)
    omega = (2.0 * np.pi * k) / N
    coeff = 2.0 * np.cos(omega)
    s_prev, s_prev2 = 0, 0
    for sample in samples:
        s = sample + coeff * s_prev - s_prev2
        s_prev2, s_prev = s_prev, s
    return s_prev2**2 + s_prev**2 - coeff*s_prev*s_prev2

# =========================
# DECODER (HARF HARF ÇALAN VERSİYON)
# =========================
def decode_audio_step_by_step(filename):
    if not filename.endswith(".wav"): filename += ".wav"
    try:
        fs_read, data = read(filename)
    except FileNotFoundError:
        print("❌ Hata: Dosya bulunamadı!")
        return

    if len(data.shape) > 1: data = data[:,0]

    window_size = int(fs_read * tone_duration)
    step_size = int(fs_read * (tone_duration + silence_duration))
    
    print(f"\n🔍 {filename} deşifre ediliyor...")
    print("📝 Metin: ", end="", flush=True) # Harfleri yan yana yazmak için

    decoded_text = ""
    last_char = None

    for start in range(0, len(data)-window_size, step_size):
        segment = data[start:start+window_size]
        
        # Sessizliği atla
        if np.max(np.abs(segment)) < threshold: continue
        
        # O harfin sesini tam o anda çal (0.1 sn yavaşlatılmış efekt için bekleyerek)
        sd.play(segment, fs_read)
        
        # Analiz (Goertzel)
        segment_ham = segment * np.hamming(len(segment))
        low_powers = [goertzel(segment_ham, lf, fs_read) for lf in low_freqs]
        high_powers = [goertzel(segment_ham, hf, fs_read) for hf in high_freqs]

        detected_low = low_freqs[np.argmax(low_powers)]
        detected_high = high_freqs[np.argmax(high_powers)]
        char = reverse_mapping.get((detected_low, detected_high), "")

        if char != last_char:
            print(char, end="", flush=True) # Harfi anında ekrana basar
            decoded_text += char
            last_char = char
            
        # Harfler arası çok hafif bir bekleme (Yavaş okuma hissi için)
        time.sleep(0.15) 

    print("\n\n✅ İşlem tamamlandı.")
    return decoded_text

# =========================
# MAIN MENU
# =========================
if __name__ == "__main__":
    while True:
        print("\n--- DTMF HARF SİSTEMİ (ADIM ADIM) ---")
        print("1 - Metni Sese Çevir ve Kaydet")
        print("2 - Kayıtlı Dosyayı Oku (Yavaş ve Sesli)")
        print("3 - Çıkış")
        
        choice = input("Seçiminiz: ")

        if choice == "1":
            text = input("Mesajınızı yazın: ")
            fname = input("Dosya adı (örn: odev1): ")
            encode_text(text, fname)
        
        elif choice == "2":
            fname = input("Okunacak dosya adı (örn: odev1): ")
            decode_audio_step_by_step(fname)
        
        elif choice == "3":
            print("Güle güle!")
            break3