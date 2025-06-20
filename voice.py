import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

# 1) Kendi konvolüsyon fonksiyonumuz (manuel):
def myConv(x, h):
    """
    x: Ses sinyali veya herhangi bir dizi
    h: Sistem cevabı veya başka bir dizi
    return: x * h konvolüsyonu
    """
    Lx = len(x)
    Lh = len(h)
    # Konvolüsyon sonucu uzunluğu Lx + Lh - 1 olacaktır
    Lz = Lx + Lh - 1
    # Çıktı vektörünü sıfırla
    z = np.zeros(Lz)
    # Konvolüsyon hesaplaması
    for i in range(Lx):
        for j in range(Lh):
            z[i + j] += x[i] * h[j]
    return z

# 2) Hazır conv fonksiyonu ile karşılaştırılması

# Örnek veri setleri
X1 = [1, 2, 3]
Y1 = [2, 1, 0]

X2 = [1, -1, 2, 4]
Y2 = [3, 3, 1]

# myConv sonuçları
myConv_result_1 = myConv(X1, Y1)
myConv_result_2 = myConv(X2, Y2)

# Hazır fonksiyon sonuçları (numpy convolve)
npConv_result_1 = np.convolve(X1, Y1)
npConv_result_2 = np.convolve(X2, Y2)

print("----- Veri Seti 1 -----")
print("X1 =", X1)
print("Y1 =", Y1)
print("myConv_result_1 =", myConv_result_1)
print("npConv_result_1 =", npConv_result_1.tolist())  # numpy array -> liste

print("\n----- Veri Seti 2 -----")
print("X2 =", X2)
print("Y2 =", Y2)
print("myConv_result_2 =", myConv_result_2)
print("npConv_result_2 =", npConv_result_2.tolist())

# Grafiksel gösterim
# 4 ayrı işareti alt alta çizelim: X, Y, myConv, np.convolve
plt.figure(figsize=(8,8))

plt.subplot(4,1,1)
plt.stem(X1)
plt.title("X1")

plt.subplot(4,1,2)
plt.stem(Y1)
plt.title("Y1")

plt.subplot(4,1,3)
plt.stem(myConv_result_1)
plt.title("MyConv Sonucu (Veri Seti 1)")

plt.subplot(4,1,4)
plt.stem(npConv_result_1)
plt.title("Hazır Conv Sonucu (Veri Seti 1)")

plt.tight_layout()
plt.show()

# İkinci veri seti için de benzer şekilde çizimler yapılabilir
plt.figure(figsize=(8,8))

plt.subplot(4,1,1)
plt.stem(X2)
plt.title("X2")

plt.subplot(4,1,2)
plt.stem(Y2)
plt.title("Y2")

plt.subplot(4,1,3)
plt.stem(myConv_result_2)
plt.title("MyConv Sonucu (Veri Seti 2)")

plt.subplot(4,1,4)
plt.stem(npConv_result_2)
plt.title("Hazır Conv Sonucu (Veri Seti 2)")

plt.tight_layout()
plt.show()



# 3) Impuls yanıtı (h[n]) oluşturan fonksiyon
#    h[n] = 1 + A + A^2 + ... + A^(M-1)  (delta'larla kayma mantığını basit dizi olarak temsil ediyoruz)
def create_h(M, A=0.5):
    """
    M: M değeri (3, 4, 5 gibi)
    A: katsayı (0.5)
    """
    # M boyutlu bir dizi: [1, A, A^2, ..., A^(M-1)]
    h = [A**i for i in range(M)]
    return np.array(h, dtype=float)

# 3) Ses kaydı fonksiyonu (5 saniyelik)
def record_audio(duration=5, fs=16000):
    """
    duration: kaç saniye kayıt yapılacağı
    fs: örnekleme frekansı (Hz)
    return: NumPy array (mono)
    """
    print("Kayıt başlıyor. Konuşun...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # kayıt bitene kadar blokla
    print("Kayıt tamamlandı.")
    return recording[:, 0]

# ---- UYGULAMA KODLARI ----

# (A) 5 saniyelik ses kaydı alalım
fs = 16000  # örnekleme frekansı
X1 = record_audio(duration=5, fs=fs)  # 5 saniye kayıt

print("X1 boyutu:", len(X1))
print("Örnek veri [ilk 10 örnek]:", X1[:10])

# (B) Farklı M değerleri için h[n] oluşturup konvolüsyon yapalım
M_values = [3, 4, 5]
A = 0.5

for M in M_values:
    print(f"\n--- M = {M} için işlem yapılıyor ---")
    hM = create_h(M, A=A)     # Sistemin impuls cevabı
    
    # (B1) Kendi konvolüsyon fonksiyonumuzla
    myY1 = myConv(X1, hM)
    
    # (B2) Hazır fonksiyonla
    Y1 = np.convolve(X1, hM)
    
    print(f"hM = {hM}")
    print(f"myY1 (ilk 10 örnek) = {myY1[:10]}...")
    print(f"Y1   (ilk 10 örnek) = {Y1[:10]}...")
    
    # İstersek konvolüsyon sonuçlarını ses olarak dinleyebiliriz:
    sd.play(myY1, fs)
    sd.wait()
    
    
    # grafiksel olarak karşılaştırma (kısa sinyallerde)
    # Ancak ses çok uzun olabileceği için grafiğe tamamını çizmek pratik olmayabilir.
    # Yine de küçük bir bölümünü çizerek örnek yapalım:
    plt.figure()
    plt.plot(myY1[:500], label='myConv (ilk 500 örnek)')
    plt.plot(Y1[:500], label='np.convolve (ilk 500 örnek)', linestyle='--')
    plt.title(f"M = {M}, Konvolüsyon Karşılaştırması")
    plt.xlabel("Zaman örnekleri")
    plt.ylabel("Genlik")
    plt.legend()
    plt.show()

