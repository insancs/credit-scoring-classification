<h1 align="center"> Credit Scoring Classification </h1>
<p align="center">
    <img src="https://miro.medium.com/max/1400/1*BIZJkKcNp8U0XBf8UX34bQ.png" width="900" height="400">
</p>

## Problem Statement

### Context
Departemen kredit konsumen sebuah bank ingin mengotomatisasi proses pengambilan keputusan untuk persetujuan jalur kredit ekuitas rumah. Untuk melakukan ini, mereka akan mengikuti rekomendasi dari Equal Credit Opportunity Act untuk membuat model penilaian kredit yang diturunkan secara empiris dan sehat secara statistik. Model ini akan didasarkan pada data yang dikumpulkan dari pemohon yang baru saja diberikan kredit melalui proses penjaminan pinjaman saat ini. Model akan dibangun dari alat pemodelan prediktif, tetapi model yang dibuat harus cukup dapat diinterpretasikan untuk memberikan alasan untuk setiap tindakan yang merugikan (penolakan).

### Dataset
Dataset Ekuitas Rumah (HMEQ) berisi informasi dasar dan kinerja pinjaman untuk 5.960 pinjaman ekuitas rumah baru-baru ini. Target (BAD) adalah variabel biner yang menunjukkan apakah pelamar akhirnya gagal atau benar-benar tunggakan. Hasil buruk ini terjadi pada 1.189 kasus (20%). Untuk setiap pelamar, 12 variabel input dicatat.
Dataset berasal dari Kaggle yang bisa dilihat [disini](https://www.kaggle.com/ajay1735/hmeq-data/code).

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

## Sample Dataset
| BAD 	| LOAN 	| MORTDUE  	| VALUE    	| REASON  	| JOB     	| YOJ  	| DEROG 	| DELINQ 	| CLAGE        	| NINQ 	| CLNO 	| DEBTINC      	|
|-----	|------	|----------	|----------	|---------	|---------	|------	|-------	|--------	|--------------	|------	|------	|--------------	|
| 1   	| 1100 	| 25860    	| 39025    	| HomeImp 	| Other   	| 10.5 	| 0     	| 0      	| 94.366666667 	| 1    	| 9    	|              	|
| 1   	| 1300 	| 70053    	| 68400    	| HomeImp 	| Other   	| 7    	| 0     	| 2      	| 121.83333333 	| 0    	| 14   	|              	|
| 1   	| 1500 	| 13500    	| 16700    	| HomeImp 	| Other   	| 4    	| 0     	| 0      	| 149.46666667 	| 1    	| 10   	|              	|
| 0   	| 1700 	| 97800    	| 112000   	| HomeImp 	| Office  	| 3    	| 0     	| 0      	| 93.333333333 	| 0    	| 14   	|              	|
| 1   	| 1700 	| 30548    	| 40320    	| HomeImp 	| Other   	| 9    	| 0     	| 0      	| 101.46600191 	| 1    	| 8    	| 37.113613558 	|


## Work Steps
Pada project ini terdapat beberapa tahapan yang dilakukan, yaitu :
1. Import Libraries
2. Data Exploration
    - Load Dataset
    - Data Information
    - Statistics Description
    - Check Missing Values
    - Check Duplicated Data
3. Data Cleansing
    - Check Cardinality
    - Handling Missing Values
    - Handling Outliers
    - Data Type Transformation
4. Exploratory Data Analysis
5. Feature Engineering
    - Feature Encoding
    - Feature Scaling
6. Sampling Data
    - Separate Train and Test Set
    - Imbalanced Class
7. Modelling
    - Choose Best Classifier
    - Tuning Hyperparameters
8. Model Performance Evaluation
    - Confusion Matrix
    - Classification Report
    - Feature Importance
9. Save Model

## Model Performance 
Pada tahap ini dilakukan uji coba pada beberapa model klasifikasi, lalu dibanding mana yang mempunyai performa terbaik untuk nentinya dilakukan tuning hyperparameters. Berikut performa pada beberapa model yang telah dilakukan training:
|   	|                     Method 	| accuracy 	| precision 	|   recall 	| f1-score 	|
|--:	|---------------------------:	|---------:	|----------:	|---------:	|----------	|
| 0 	|       ExtraTreesClassifier 	| 0.937919 	|  0.936364 	| 0.774436 	| 0.847737 	|
| 1 	|       KNeighborsClassifier 	| 0.926174 	|  0.873950 	| 0.781955 	| 0.825397 	|
| 2 	|     RandomForestClassifier 	| 0.918904 	|  0.843243 	| 0.781955 	| 0.811443 	|
| 3 	|          BaggingClassifier 	| 0.883110 	|  0.755376 	| 0.704261 	| 0.728923 	|
| 4 	| GradientBoostingClassifier 	| 0.874161 	|  0.703271 	| 0.754386 	| 0.727932 	|
| 5 	|     DecisionTreeClassifier 	| 0.862975 	|  0.695431 	| 0.686717 	| 0.691047 	|
| 6 	|         AdaBoostClassifier 	| 0.838367 	|  0.614583 	| 0.739348 	| 0.671217 	|
| 7 	|                        SVC 	| 0.794183 	|  0.528131 	| 0.729323 	| 0.612632 	|
| 8 	|                 GaussianNB 	| 0.776286 	|  0.498471 	| 0.408521 	| 0.449036 	|
| 9 	|         LogisticRegression 	| 0.737696 	|  0.440678 	| 0.651629 	| 0.525784 	|

Model ExtraTreesClassifier memiliki performa terbaik dengan akurasi, precision, recall, dan f1-score tertinggi. Maka model ini yang akan dilakukan tuning hyperparameters.

Setelah dilakukan tuning hyperparameters, didapatkan performa pada model :
### Confusion Matrix 

|       	| pred 0 	| pred 1 	|
|------:	|-------:	|-------:	|
| act 0 	|   1370 	|     19 	|
| act 1 	|     85 	|    314 	|

### Classification Report 

                  precision    recall  f1-score   support

               0       0.94      0.99      0.96      1389
               1       0.94      0.79      0.86       399

        accuracy                           0.94      1788
        macro avg      0.94      0.89      0.91      1788
        weighted avg   0.94      0.94      0.94      1788

### Feature Importance
<p align="left">
    <img src="feature importance.png" width="700" height="500">
</p>


## Summary
### Korelasi Feature dan Target
- Pada data yang dilatih class yang dominan yaitu class 0 (pemohon berhasil membayar)
- Distribusi jumlah pinjaman antara class 0 dan class 1 memiliki distribusi yang sama, maka bisa dikatakan jumlah pinjaman tidak berpengaruh kepada pemohon gagal membayar
- Distribusi pemohon yang gagal membayar dan tidak cenderung sama pada pada setiap alasan peminjaman, jadi bisa dikatakan alasan peminjaman tidak berpengaruh kepada gagalnya pembayaran pinjaman
- Jumlah laporan penghinaan utama, jumlah kredit menunggak cukup tinggi memiliki korelasi positif yang cukup tinggi pada faktor kegagalan pemohon membayar. Karena secara umum semakin banyak kredit mengunggak maka semakin sulit juga untuk membayar.

### Model Performance
- Pada hasil percobaan didapatkan bahwa model Extra Tree Classifier memiliki performa terbaik dari beberapa model yang digunakan, dengan akurasi mencapai 93.79%.
- Dilakukan tuning hyperparameter untuk memilih parameter terbaik yang akan digunakan dalam training model.
- Dari hasil confusion matrix didapatkan TP dan TN yang tinggi, maka bisa di katakan model dapat mengklasifikasi dengan cukup baik dengan akurasi mencapai 94%. 
