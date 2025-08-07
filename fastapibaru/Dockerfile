
# Gunakan image Python resmi sebagai base image
FROM python:3.9-slim

# Atur direktori kerja di dalam kontainer
WORKDIR /app

# Salin file requirements.txt (jika ada) dan install dependensi
# Asumsikan ada file requirements.txt di root folder fastapibaru
COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

# Salin seluruh isi folder aplikasi FastAPI ke direktori kerja di kontainer
COPY ./app/api/main.py /app/api/main.py

# Salin folder model ke dalam kontainer
COPY ./modelbaru /app/modelbaru
# Ekspos port yang akan digunakan oleh FastAPI (sesuai dengan yang di-bind di docker-compose)
EXPOSE 80

# Perintah untuk menjalankan aplikasi FastAPI menggunakan uvicorn
# Host 0.0.0.0 agar bisa diakses dari luar kontainer Docker
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "80"]
