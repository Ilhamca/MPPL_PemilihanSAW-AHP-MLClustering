# Sistem Pendukung Keputusan Multi-Kriteria Pemilihan Laptop Mahasiswa Berbasis Web dengan Metode Hybrid (AHP + SAW) dan Integrasi Machine Learning

## Perencanaan:
a. Implementasi Metode Hybrid: Menerapkan metode gabungan Analytical Hierarchy Process (AHP) untuk proses pembobotan kriteria secara sistematis dan objektif, serta metode Simple Additive Weighting (SAW) untuk melakukan proses perankingan alternatif laptop secara efisien dan akurat.  
b. Integrasi Machine Learning: Mengintegrasikan algoritma machine learning, yaitu clustering dengan metode K-Means, untuk menganalisis dan mengelompokkan dataset laptop secara otomatis ke dalam kategori-kategori yang relevan dengan tipe pengguna mahasiswa (misalnya: kebutuhan desain grafis, programming, atau penggunaan umum).  
c. Pengembangan Aplikasi Web: Merancang dan mengembangkan platform aplikasi berbasis web yang responsif dan mudah diakses, yang berfungsi sebagai antarmuka utama bagi pengguna untuk berinteraksi dengan sistem.  
d. Visualisasi Hasil yang Interaktif: Menyajikan hasil akhir rekomendasi kepada pengguna dalam bentuk visualisasi data yang informatif dan mudah dipahami, mencakup daftar peringkat laptop, grafik perbandingan antar kriteria, serta penandaan kategori laptop berdasarkan hasil clustering.

## Batasan Teknis dan Platform

- Platform: Sistem akan dikembangkan sebagai aplikasi berbasis web agar mudah diakses melalui peramban tanpa perlu instalasi.  
- Teknologi: Pembangunan sistem akan menggunakan bahasa pemrograman Python dengan library Scikit-learn untuk implementasi machine learning. Untuk kerangka kerja (framework) web, akan dipilih antara Streamlit (untuk pengembangan cepat) atau Django (untuk aplikasi yang lebih skalabel).  
- Batasan Fungsional: Aplikasi ini berfokus pada pemberian rekomendasi dan tidak mencakup fitur transaksi pembelian atau pengecekan stok produk secara real-time dari marketplace.