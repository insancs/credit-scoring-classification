<h1 align="center"> Credit Scoring Classification </h1>

## Problem Statement
### Departemen kredit konsumen sebuah bank ingin mengotomatisasi proses pengambilan keputusan untuk persetujuan jalur kredit ekuitas rumah. Untuk melakukan ini, mereka akan mengikuti rekomendasi dari Equal Credit Opportunity Act untuk membuat model penilaian kredit yang diturunkan secara empiris dan sehat secara statistik. Model ini akan didasarkan pada data yang dikumpulkan dari pemohon yang baru saja diberikan kredit melalui proses penjaminan pinjaman saat ini. Model akan dibangun dari alat pemodelan prediktif, tetapi model yang dibuat harus cukup dapat diinterpretasikan untuk memberikan alasan untuk setiap tindakan yang merugikan (penolakan).

### Content
Dataset Ekuitas Rumah (HMEQ) berisi informasi dasar dan kinerja pinjaman untuk 5.960 pinjaman ekuitas rumah baru-baru ini. Target (BAD) adalah variabel biner yang menunjukkan apakah pelamar akhirnya gagal atau benar-benar tunggakan. Hasil buruk ini terjadi pada 1.189 kasus (20%). Untuk setiap pelamar, 12 variabel input dicatat.

### Goals
Memprediksi apakah seseorang akan gagal membayar pinjaman atau tidak, berdasarkan atribut yang diberikan.

### Attribute Overview
- BAD: 1 = pemohon gagal dalam pinjaman atau tunggakan yang serius; 0= pinjaman yang dibayar pemohon (Variabel Target){Nominal Biner Assimetris}
- LOAN: Jumlah permintaan pinjaman{Rasio-Skala Numerik}
- MORTDUE: Jumlah yang harus dibayar pada hipotek yang ada{Rasio-Skala Numerik}
- VALUE: Nilai properti saat ini{Rasio-Skala Numerik}
- REASON: DebtCon = konsolidasi hutang; homelmp = Perbaikan rumah{Nominal}
- JOB: Kategori pekerjaan{Nominal}
- YOJ: Tahun di pekerjaan sekarang
- DEROG: Jumlah laporan penghinaan utama
- DELINQ: Jumlah kredit yang menunggak
- CLAGE: Usia batas kredit tertua dalam beberapa bulan
- NINQ: Jumlah pertanyaan kredit terbaru
- CLNO: Jumlah jalur kredit
- DEBTINC: Rasio utang terhadap pendapatan{Rasio-Skala Numerik}

