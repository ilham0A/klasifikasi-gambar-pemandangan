# Klasifikasi Gambar Pemandangan: Intel Image Classification

Proyek akhir kelas Fundamental Deep Learning. Model CNN (Sequential, dibangun dari awal tanpa model pretrained) untuk mengklasifikasikan gambar pemandangan ke dalam 6 kelas, lalu diekspor ke format SavedModel, TF-Lite, dan TFJS.

**Penulis:** Muhammad Ilham

## Dataset

- **Sumber:** [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) (Kaggle, `puneet6060/intel-image-classification`).
- **Kelas (6):** `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`.
- **Jumlah gambar:** 17.034 pada dataset asli (14.034 di `seg_train` dan 3.000 di `seg_test`). Setelah seluruh gambar digabung, 18 gambar duplikat persis dibuang sehingga tersisa **17.016 gambar** dengan jumlah per kelas yang relatif seimbang (2.627 sampai 3.031 gambar per kelas).
- **Resolusi:** ada 32 ukuran berbeda, tetapi 16.979 gambar (99,68%) berukuran 150x150. Gambar lainnya umumnya berlebar 150 dengan tinggi lebih pendek. Jadi dataset ini tidak sepenuhnya seragam, namun variasinya kecil. Semua gambar diubah ke 128x128 hanya di pipeline pemuatan data; berkas aslinya tidak diubah.

## Pembagian Data

Dataset bawaan sudah terbagi menjadi `seg_train` dan `seg_test`, tetapi pembagiannya dilakukan ulang secara manual:

1. Seluruh gambar dari `seg_train` dan `seg_test` digabung ke satu folder per kelas, dan gambar duplikat persis dibuang (hash MD5 isi berkas).
2. Setiap kelas dibagi acak (seed 42) menjadi train 80%, validation 10%, dan test 10%, sehingga proporsi kelas di tiap bagian tetap seimbang.
3. Hash seluruh berkas di ketiga bagian dibandingkan, dan notebook memastikan tidak ada gambar yang sama di lebih dari satu bagian.

| Bagian     | Jumlah | Proporsi |
| ---------- | ------ | -------- |
| Train      | 13.609 | 80,0%    |
| Validation | 1.702  | 10,0%    |
| Test       | 1.705  | 10,0%    |

## Model

Model `Sequential` dengan `Conv2D` dan `MaxPooling2D`:

- Augmentasi di dalam model (`RandomFlip` horizontal, `RandomRotation(0.05)`, `RandomZoom(0.1)`), hanya aktif saat training sehingga validation dan test tidak diaugmentasi.
- `Rescaling(1/255)`.
- Empat blok konvolusi: 2x `Conv2D(32)`, 2x `Conv2D(64)`, 2x `Conv2D(128)` (masing-masing diikuti `BatchNormalization` dan diakhiri `MaxPooling2D`), lalu `Conv2D(256)`.
- `GlobalAveragePooling2D`, `Dense(256)`, `Dropout(0.4)`, dan `Dense(6, softmax)`.

**Training:** Adam (learning rate 1e-3), batch 32, input 128x128, maksimal 60 epoch. Callback yang dipakai:

- `ReduceLROnPlateau`: learning rate dikali 0,5 jika `val_loss` tidak membaik selama 3 epoch.
- `EarlyStopping`: berhenti setelah 12 epoch tanpa perbaikan `val_loss`, lalu mengembalikan bobot terbaik.
- `StopAt95` (callback khusus): berhenti jika akurasi train dan validation sama-sama mencapai 95%.

## Hasil

| Metrik                 | Nilai                                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------ |
| Akurasi train          | 92,17%                                                                                     |
| Akurasi validation     | 89,31%                                                                                     |
| Akurasi test           | 88,80%                                                                                     |
| Akurasi test (TF-Lite) | 88,80% (1.514 dari 1.705 gambar benar)                                                     |
| Epoch yang dijalankan  | 31 dari maksimal 60 (dihentikan `EarlyStopping`; bobot terbaik dari epoch 19 dikembalikan) |

Akurasi train diukur setelah training pada gambar asli (tanpa augmentasi). Akurasi TF-Lite sama dengan akurasi model Keras pada test set, sehingga konversi tidak mengubah performa model.

**Performa per kelas (test set):**

| Kelas     | Precision | Recall | F1-score |
| --------- | --------- | ------ | -------- |
| buildings | 0,89      | 0,89   | 0,89     |
| forest    | 0,97      | 0,97   | 0,97     |
| glacier   | 0,84      | 0,83   | 0,83     |
| mountain  | 0,82      | 0,86   | 0,84     |
| sea       | 0,91      | 0,87   | 0,89     |
| street    | 0,91      | 0,91   | 0,91     |

Plot akurasi dan loss, classification report, dan confusion matrix ada di `notebook.ipynb`.

**Catatan:** kesalahan prediksi terbanyak terjadi pada pasangan kelas yang mirip secara visual: 38 gambar `glacier` diprediksi sebagai `mountain` (dan 31 sebaliknya), 21 gambar `buildings` diprediksi sebagai `street` (dan 17 sebaliknya), serta 17 gambar `sea` diprediksi sebagai `glacier`. Akurasi test (88,80%) belum mencapai 95%; model dibangun dari awal (tanpa transfer learning) sesuai ketentuan kriteria.

## Konversi Model

- **SavedModel:** diekspor dari salinan model tanpa layer augmentasi (bobot sama), karena augmentasi hanya diperlukan saat training. Selain `saved_model.pb` dan `variables/`, TensorFlow otomatis membuat folder `assets/` (kosong karena model tidak memakai berkas tambahan) dan `fingerprint.pb`.
- **TF-Lite:** dikonversi dari SavedModel; urutan label di `label.txt` sama dengan indeks kelas model.
- **TFJS:** `tensorflowjs_converter` 4.22.0 (format graph model) dijalankan di lingkungan Python terpisah dari SavedModel.

## Inference

Inference memakai model TF-Lite pada seluruh test set, dan delapan gambar contoh ditampilkan beserta prediksi dan label aslinya (lihat bagian Inference di notebook).

## Struktur Direktori

```
submission
├── tfjs_model
│   ├── group1-shard1of1.bin
│   └── model.json
├── tflite
│   ├── model.tflite
│   └── label.txt
├── saved_model
│   ├── saved_model.pb
│   ├── fingerprint.pb
│   └── variables
│   └── assets
├── Submission_Klasifikasi_Gambar.ipynb
├── README.md
└── requirements.txt
```

## Menjalankan Ulang

1. Buka `Submission_Klasifikasi_Gambar.ipynb` di Google Colab dan pilih runtime **T4 GPU**.
2. Jalankan semua sel dari atas ke bawah. Dataset diunduh otomatis lewat `kagglehub`.
3. Training memakan waktu sekitar 15 sampai 30 menit; konversi TFJS sekitar 3 sampai 8 menit.
4. Versi paket yang dipakai ada di `requirements.txt`. Paket `tensorflowjs` hanya dicatat sebagai komentar karena notebook memasangnya sendiri di lingkungan Python terpisah (versi 4.22.0 tidak bisa dipasang bersama `tensorflow==2.21.0` di lingkungan yang sama).

## 🤖 AI Attribution / Acknowledgements

- **Ide & Konsep:** Menggunakan Claude untuk brainstorming pemilihan dataset dan strategi mencapai target akurasi saat Animals-10 (5 kelas) hanya mencapai sekitar 55% pada uji kelayakan. Hasilnya, dataset dialihkan ke Intel Image Classification dan arsitektur CNN diperbaiki (dua `Conv2D` per blok, `GlobalAveragePooling2D`, serta callback `ReduceLROnPlateau`, `EarlyStopping`, dan `StopAt95`).
- **Cell Inference:** Dibantu Claude untuk menyusun fungsi `predict_tflite()` dan sel yang mengukur akurasi model TF-Lite pada seluruh test set, serta menampilkan delapan gambar contoh beserta prediksi dan label aslinya.
- Dibantu Claude untuk menyusun README.md
