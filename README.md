# TİD Asistanı (Türk İşaret Dili Asistanı)

TİD Asistanı, Türk İşaret Dili'ni (TİD) dijital ortamda daha erişilebilir kılmak, işaret dili öğrenimini desteklemek ve sağır/işitme engelli bireylerle iletişim bariyerlerini ortadan kaldırmak amacıyla geliştirilen bir yapay zeka destekli yardımcı araçtır.

## 🎯 Projenin Amacı

Bu projenin temel vizyonu, MediaPipe ve derin öğrenme tekniklerini kullanarak Türk İşaret Dili hareketlerini gerçek zamanlı olarak tanımak ve metne/sese dönüştürmektir.

## 🚀 Mevcut Durum (Current Stage)

Proje şu an **Veri Toplama ve Ön İşleme (Phase 1)** aşamasındadır.

- **Veri Toplama Aracı:** MediaPipe kullanarak el verilerini (landmarks) 3D koordinat sisteminde (x, y, z) kaydeden fonksiyonel bir istemci uygulaması hazırlandı.
- **Normalizasyon:** Elin kameraya uzaklığından bağımsız olması için bilek merkezli normalizasyon algoritması uygulandı.
- **Veri Yapısı:** Harf ve kelime eğitimleri için JSON formatında yapılandırılmış veri setleri oluşturuluyor.
- **Backend:** Kullanıcı ve veri yönetimi için Prisma tabanlı temel veritabanı mimarisi kuruldu.

## 🛠️ Kurulum ve Çalıştırma

### Gereksinimler

- Node.js (v18+)
- Postgres (Prisma üzerinden otomatik yapılandırılır)

### Adımlar

#### 1. Projeyi Klonlayın

```bash
git clone [repository-url]
cd "Tid Asistan"
```

#### 2. Backend Kurulumu

```bash
cd backend
npm install
npx prisma generate
npx prisma migrate dev  # Veritabanını oluşturur
npm run dev
```

#### 3. Client (İstemci) Kurulumu

```bash
cd ../client
npm install
npm run dev
```

## 📈 Gelecek Planları

1. **Model Eğitimi:** Toplanan verilerle ilk harf ve kelime tahminleme modellerinin (LSTM/RNN) eğitilmesi.
2. **Vücut Takibi:** Kelimelerin daha doğru tanınması için omuz ve baş pozisyonlarının sisteme dahil edilmesi.
3. **Gerçek Zamanlı Çeviri:** Canlı kamera görüntüsü üzerinden anlık TİD - Türkçe metin çevirisi.

---

_Bu proje geliştirme aşamasındadır._
