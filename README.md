# Workflow-CI — Indonesian Emotion Classification

Repository CI untuk **Kriteria 3** submission MSML — otomatis re-training model IndoBERT setiap kali trigger dipantik.

## 📁 Struktur Folder

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                         # GitHub Actions CI (Advanced)
├── MLProject/
│   ├── modelling.py                       # Script training
│   ├── conda.yaml                         # Environment definition
│   └── MLProject                          # MLflow Project config
├── twitter_emotion_preprocessing/         # Dataset siap training
│   ├── train.csv
│   ├── val.csv
│   ├── full.csv
│   ├── label_encoder.pkl
│   └── metadata.json
└── README.md
```

## ⚙️ GitHub Actions Workflow Steps

| Step | Deskripsi |
|---|---|
| Set up job | Print info trigger, branch, commit |
| Run actions/checkout@v3 | Checkout repository |
| Set up Python 3.12.7 | Setup Python environment |
| Check Env | Verifikasi Python, pip, disk, memory |
| Install dependencies | Install semua library yang dibutuhkan |
| Set MLflow Tracking URI | Konfigurasi DagsHub sebagai remote tracking |
| Run mlflow project | Jalankan `mlflow run` dengan CI_MODE=true |
| Get latest MLflow run_id | Ambil run ID terbaru dari DagsHub |
| Install Python dependencies | Install library untuk upload |
| Upload to GitHub | Simpan artefak ke GitHub Actions |
| Build Docker Model | Build image dari MLflow model |
| Log in to Docker Hub | Autentikasi ke Docker Hub |
| Tag Docker Image | Tag image dengan SHA commit |
| Push Docker Image | Push ke Docker Hub |
| Complete job | Print CI summary |

## 🔐 Secrets yang Dibutuhkan

Tambahkan secrets berikut di **GitHub repo → Settings → Secrets → Actions**:

| Secret | Deskripsi |
|---|---|
| `DAGSHUB_USERNAME` | Username DagsHub kamu |
| `DAGSHUB_REPO_NAME` | Nama repo DagsHub |
| `DAGSHUB_TOKEN` | Access token DagsHub |
| `DOCKERHUB_USERNAME` | Username Docker Hub |
| `DOCKERHUB_TOKEN` | Access token Docker Hub |

## 🚀 Cara Trigger

### Otomatis
Push ke branch `main` yang mengubah file di `MLProject/` atau dataset.

### Manual
1. Buka tab **Actions** di GitHub
2. Pilih workflow **CI — IndoBERT Emotion Classification**
3. Klik **Run workflow**
4. Isi parameter opsional (sample size, epochs)
5. Klik **Run workflow**

## 🐳 Docker Image

Setelah CI berhasil, Docker image tersedia di Docker Hub:
```bash
docker pull <DOCKERHUB_USERNAME>/indonesian-emotion-classifier:latest

# Jalankan serving
docker run -p 5001:8080 <DOCKERHUB_USERNAME>/indonesian-emotion-classifier:latest
```
