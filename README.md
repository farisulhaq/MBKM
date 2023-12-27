# Aplikasi Sistem Rekomendasi Film Berbasis Hybrid Kombinasi Metode User-based dan Item-based


<details>

<!-- ## Table of Contents -->

<summary>Table of Contents</summary> 

- [Description](#description)
- [Installtion](#installtion)
  - [Clone the project](#clone-the-project)
  - [Go to the project directory](#go-to-the-project-directory)
  - [Install dependencies](#install-dependencies)
- [Tests](#tests)
- [Demo](#demo)

</details>

<hr>

## Description

<div align="justify">
Aplikasi Sistem Rekomendasi Film Berbasis Hybrid Kombinasi Metode User-based dan Item-based – Aplikasi ini adalah prototipe yang mensimulasikan kerja dan analisa dari sistem rekomendasi. Hybrid yang dimaksud yaitu mengkombinasikan antara metode User-based Collaborative Filtering (UCF) dan metode Item-based Collaborative Filtering (ICF) dengan menggunakan kombinasi similarity function Pearson Correlation Coefficient (PCC), atau Adjusted Mutual Information (AMI) yang diterapkan pada dataset MovieLens 100K. Pengguna aplikasi dapat melihat hasil rekomendasi sesuai dengan jumlah top-n rekomendasi dari user target yang dipilih serta dapat melihat evaluasi berdasarkan metrik Recall, Precision, Discounted Cumulative Gain (DCG), dan Normalized Discounted Cumulative Gain (NCDG) dari hasil rekomendasi.
</div>

<hr>

## Installtion

### Clone the project

```bash
  git clone https://github.com/farisulhaq/MBKM.git
```

### Go to the project directory

```bash
  cd MBKM
```

### Install dependencies

```bash
  python -m venv venv
  .\venv\Scripts\activate or venv/Scripts/activate
  pip install -r requirements.txt
```


## Tests
before performing the tests, please check the path variable in the app.py file.

To run tests, run the following command

```bash
  python -m flask run
```

## Demo

[farsulhaq.pythonanywhere.com](https://farsulhaq.pythonanywhere.com/)