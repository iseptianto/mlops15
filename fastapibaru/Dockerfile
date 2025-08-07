
# Gunakan image Python resmi sebagai base image
FROM python:3.9-slim

# Atur direktori kerja di dalam kontainer
WORKDIR /app

# Salin file requirements.txt (jika ada) dan install dependensi
# Asumsikan ada file requirements.txt di root folder fastapibaru
COPY ./mlflow_server /mlflow_server/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

# Salin seluruh isi folder aplikasi FastAPI ke direktori kerja di kontainer
COPY ./app/api/main.py /app/api/main.py

# Salin folder model ke dalam kontainer
COPY ./app /app/
# COPY /modelbaru/place_id_map.pkl 
# COPY /modelbaru/prediction_matrix_best.pkl 
# COPY /modelbaru/user_id_map.pkl 

# Ekspos port yang akan digunakan oleh FastAPI (sesuai dengan yang di-bind di docker-compose)
EXPOSE 80

# Perintah untuk menjalankan aplikasi FastAPI menggunakan uvicorn
# Host 0.0.0.0 agar bisa diakses dari luar kontainer Docker
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "80"]
