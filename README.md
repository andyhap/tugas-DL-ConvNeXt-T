# ConvNeXt-Tiny Imagenette — Google Colab

## Target runtime
PyTorch 2.11.0+cu130, Torchvision 0.26.0+cu130, Tesla T4.

## Penggunaan
1. Jalankan instalasi dan cek GPU pada notebook.
2. Ekstrak Imagenette ke `/content/imagenette2-160/` sehingga terdapat
   folder `train/` dan `val/`.
3. Letakkan file Python dalam direktori kerja Colab.
4. Jalankan `test_model.py`, lalu `data_pipeline.py`.

## Perubahan dari file awal
- Menggunakan `ImageFolder` untuk struktur folder dataset yang sudah diekstrak.
- Mengganti `._samples` dengan API publik `targets`.
- Menyamakan transform evaluasi dengan notebook: `Resize(img_size)` lalu
  `CenterCrop(img_size)`.
- Tidak mem-pin PyTorch/Torchvision ke versi yang berbeda dari runtime Colab.

`weights=None` berarti model dilatih dari awal. Benchmark satu epoch bukan hasil
akhir. Untuk eksperimen final, jalankan seed yang ditentukan, scheduler yang
sesuai protokol, checkpoint per epoch, dan evaluasi pada `val/` setelah pemilihan
hyperparameter dilakukan dengan validasi internal.
