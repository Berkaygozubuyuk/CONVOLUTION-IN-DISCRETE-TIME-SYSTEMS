import numpy as np
import matplotlib.pyplot as plt

def myConv(x, y):
    Lx = len(x)
    Ly = len(y)
    Lz = Lx + Ly - 1
    z = [0]*Lz
    for n in range(Lz):
        temp = 0
        for k in range(Lx):
            j = n - k
            if j >= 0 and j < Ly:
                temp += x[k] * y[j]
        z[n] = temp
    return z

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
