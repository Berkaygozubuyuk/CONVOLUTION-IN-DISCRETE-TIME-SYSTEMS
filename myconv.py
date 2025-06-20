def myConv(x, y):
    """
    x: Birinci işaret (Python listesi veya numpy array olabilir)
    y: İkinci işaret (Python listesi veya numpy array olabilir)
    
    return: x ve y'nin konvolüsyonu (liste)
    """
    Lx = len(x)
    Ly = len(y)
    # Konvolüsyon sonucu uzunluğu Lx + Ly - 1 olacaktır
    Lz = Lx + Ly - 1
    
    # Çıktı vektörünü sıfırla
    z = [0]*Lz
    
    # Konvolüsyon hesaplaması
    for i in range(Lx):
        for j in range(Ly):
            z[i + j] += x[i] * y[j]
    return z


# Örnek kontrol
x = [1, 2, 3]
y = [1, 2, 3]
sonuc_myConv = myConv(x, y)
print("myConv(x,y) =", sonuc_myConv)   # Beklenen: [1, 4, 10, 12, 9]
