# MLOps15 — Tourism Recommender (Docker Compose Stack)

Stack ini menyiapkan **MLflow + MinIO (S3) + Postgres + Trainer + FastAPI + Prometheus + Grafana + Nginx**.
Targetnya: user cukup jalankan `docker compose up --build model_tourism_trainer` untuk melatih model, lalu **FastAPI** otomatis melayani prediksi berdasarkan model di **Model Registry**.

## Isi & Arsitektur

**Service utama**

* **`mlflow_server`** — UI & Tracking Server (port 5001)
* **`db`** — Postgres untuk backend MLflow
* **`minio`** — Object store S3-compat (port 9000, console 9001)
* **`minio-mc`** — init job untuk membuat bucket `mlflow`
* **`model_tourism_trainer`** — trainer untuk *tourism recommender* (scikit-learn)
* **`fastapi_tourism_app`** — REST API model
* **`prometheus_server`** — metrics scraping (port 9090)
* **`grafana_server`** — dashboard (port 3000)
* **`nginx_load_balancer`** — reverse proxy (port 80)

**Struktur direktori (ringkas)**

```
mlops15/
├─ docker-compose.yml
├─ data/                       # taruh dataset.csv di sini (opsional)
├─ mlflow_server/              # Dockerfile + requirements MLflow server
├─ modelbaru/                  # train.py + requirements + Dockerfile (trainer)
├─ fastapibaru/                # FastAPI app + Dockerfile
├─ nginx/                      # nginx.conf
├─ prometheus/                 # prometheus.yml
└─ grafana/                    # provisioning + dashboards
```

---

## Prasyarat

* Docker & Docker Compose (v2)
* Port bebas: **80, 5001, 9000, 9001, 9090, 3000, 8101**

---

## Quick Start

> **Pertama kali** jalan, **wajib** membuat bucket S3 `mlflow` (di-*bootstrap* otomatis oleh `minio-mc` di langkah di bawah).

```bash
# 1) Matikan stack lama (jika ada)
docker compose down

# 2) Build image yang penting
docker compose build --no-cache mlflow_server model_tourism_trainer fastapi_tourism_app

# 3) Naikkan MinIO & init bucket
docker compose up -d minio minio-mc

# 4) Naikkan Postgres + MLflow
docker compose up -d db mlflow_server
docker compose logs -f mlflow_server  # tunggu "listening on 0.0.0.0:5001"

# 5) (Opsional) Siapkan data
# letakkan dataset Anda di ./data/dataset.csv dan pastikan ada kolom target "target"

# 6) Jalankan trainer (akan log ke MLflow & upload artifacts ke MinIO)
docker compose up --build model_tourism_trainer

# 7) Jalankan FastAPI + layanan monitoring
docker compose up -d fastapi_tourism_app prometheus_server grafana_server nginx_load_balancer
```

**Akses UI**

* MLflow: `http://localhost:5001`
* MinIO Console: `http://localhost:9001` (user/pass: `minioadmin/minioadmin`)
* FastAPI (direct): `http://localhost:8101/docs`
* FastAPI via Nginx: `http://localhost/`
* Prometheus: `http://localhost:9090`
* Grafana: `http://localhost:3000` (user/pass default: `admin/admin`)

---

## Alur Training → Serving

1. **Training**
   Service `model_tourism_trainer` menjalankan `modelbaru/train.py`:

   * Membaca CSV default di `/app/data/dataset.csv` (volume `./data:/app/data`).
   * Kolom target default: `target` (ubah lewat argumen).
   * Mencatat **params/metrics/artifacts** ke MLflow, serta **log model** (flavor sklearn).

2. **Register & Alias**
   FastAPI membaca model via **Model Registry alias**:

   ```
   models:/tourism-recommender-model@production
   ```

   Artinya, setelah training Anda harus **register model** dan memberi **alias `production`** pada versi terbaru.

   **Opsi A — Manual via UI**

   * MLflow UI → buka *Run* terbaru → artifact `model` → **Register model** → name: `tourism-recommender-model`.
   * Tab **Models** → `tourism-recommender-model` → pilih **Version terbaru** → **Add alias** → `production`.

   **Opsi B — Otomatis saat training**
   Set env berikut di service `model_tourism_trainer`:

   ```
   MLFLOW_REGISTERED_MODEL=tourism-recommender-model
   ```

   Tambahkan snippet berikut di **akhir** `modelbaru/train.py` (setelah `mlflow.sklearn.log_model(...)`):

   ```python
   from mlflow.tracking import MlflowClient
   reg_name = (args.registered_model_name or os.getenv("MLFLOW_REGISTERED_MODEL","")).strip()
   if reg_name:
       client = MlflowClient()
       mv = next(int(m.version) for m in client.search_model_versions(f"name='{reg_name}'")
                 if m.run_id == run.info.run_id)
       client.set_registered_model_alias(reg_name, "production", mv)
       print(f"Alias 'production' -> {reg_name} v{mv}")
   ```

3. **Serving**
   Setelah alias `production` ada, restart FastAPI agar memuat model:

   ```bash
   docker compose restart fastapi_tourism_app
   ```

---

## Konfigurasi Penting (sudah diset di Compose)

**MLflow Server**

* Backend: `postgresql+psycopg2://mlflow:mlflow@db:5432/mlflow`
* Artifact root: `s3://mlflow/`
* Env S3:

  ```
  AWS_ACCESS_KEY_ID=minioadmin
  AWS_SECRET_ACCESS_KEY=minioadmin
  MLFLOW_S3_ENDPOINT_URL=http://minio:9000
  AWS_DEFAULT_REGION=us-east-1
  AWS_EC2_METADATA_DISABLED=true
  AWS_S3_ADDRESSING_STYLE=path
  ```

**Trainer**

* Tracking URI: `http://mlflow_server:5001`
* Default args:

  ```
  --data_csv /app/data/dataset.csv
  --target target
  --task auto
  --model rf
  ```

> Ubah argumen di `docker-compose.yml` bila perlu, atau override via:
>
> ```bash
> docker compose run --rm model_tourism_trainer \
>   python train.py --data_csv /app/data/your.csv --target your_target --task auto --model rf
> ```

---

## Perintah Harian

```bash
# Lihat service yang jalan
docker compose ps

# Logs realtime
docker compose logs -f mlflow_server
docker compose logs -f model_tourism_trainer
docker compose logs -f fastapi_tourism_app

# Restart service tertentu
docker compose restart fastapi_tourism_app

# Hentikan semua (tanpa hapus data)
docker compose down

# Hentikan + hapus volume (hapus data!) — hati-hati
docker compose down -v
```

---

## Troubleshooting

**1) `alias production not found` di FastAPI / Nginx**
Penyebab: belum register model & alias.
Solusi: Lihat bagian **Register & Alias** di atas. Setelah alias dibuat, `docker compose restart fastapi_tourism_app`.

**2) MLflow gagal start (koneksi Postgres/driver)**
Pastikan image `mlflow_server` berisi paket:

```
mlflow==2.21.2
psycopg2-binary==2.9.9
boto3==1.34.131
```

`docker compose build --no-cache mlflow_server` lalu `up` lagi.

**3) `NoSuchBucket` / artifact tidak terkirim**
Pastikan service `minio-mc` sukses membuat bucket `mlflow`. Cek:

```bash
docker compose logs --tail=200 minio-mc
```

Atau buka MinIO Console ([http://localhost:9001](http://localhost:9001)) dan lihat apakah bucket `mlflow` ada.

**4) Konflik versi NumPy / MLflow saat build trainer**

* Gunakan **MLflow ≥ 2.20** agar kompatibel dengan **NumPy 2.x**.
* Contoh `modelbaru/requirements.txt` yang aman:

  ```
  mlflow==2.21.2
  pandas==2.2.2
  scikit-learn==1.5.1
  matplotlib==3.9.0
  numpy==2.0.1
  ```

**5) MLflow UI tidak muncul di 5001**

* Cek logs: `docker compose logs -f mlflow_server`
* Pastikan `db` *healthy* (ada healthcheck) dan `minio` sudah jalan.

**6) FastAPI gagal load model**

* Cek logs: `docker compose logs -f fastapi_tourism_app`
* Pastikan alias `production` ada di **Models → tourism-recommender-model**.
* Pastikan `MLFLOW_TRACKING_URI` menuju `mlflow_server` (bukan `localhost` dari dalam container).

---

## Kustomisasi Dataset & Target

* Letakkan file di `./data/dataset.csv`
* Pastikan ada kolom target; default `target`.
* Ubah target via argumen Compose:

  ```yaml
  command: >
    python train.py
    --data_csv /app/data/dataset.csv
    --target your_target_column
    --task auto
    --model rf
  ```

---

## Endpoints Penting

* **Nginx (reverse proxy)**: `http://localhost/`

  * Health / Root bisa mengembalikan JSON status model (tergantung implementasi di FastAPI).
* **FastAPI (langsung)**: `http://localhost:8101/docs`
* **MLflow UI**: `http://localhost:5001`
* **MinIO Console**: `http://localhost:9001` (login `minioadmin / minioadmin`)
* **Prometheus**: `http://localhost:9090`
* **Grafana**: `http://localhost:3000` (login `admin / admin`)

---

## Catatan

* Jangan mendefinisikan **service dengan nama sama dua kali** di `docker-compose.yml` (contoh: `minio:` dobel). Simpan **satu** saja.
* Bagian `deploy:` (resource limits) di Compose **diabaikan** di mode non-Swarm. Aman dibiarkan atau pindah ke Swarm/Kubernetes bila perlu QoS yang ketat.

---

## Lisensi

Bebas dipakai untuk belajar & pengembangan. Attribution ke repo asal diapresiasi 🙌

