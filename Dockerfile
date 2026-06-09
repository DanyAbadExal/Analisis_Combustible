# 1. Imagen base con la versión específica de Python
FROM python:3.14.3-slim-bookworm

# 2. Directorio de trabajo dentro del contenedor
WORKDIR /app

# Forzar a Debian a usar HTTPS para evitar bloqueos de red
RUN sed -i 's/http:/https:/g' /etc/apt/sources.list.d/debian.sources

# 3. Instalar dependencias del sistema y el ODBC Driver 18 para SQL Server
RUN apt-get update && apt-get install -y \
    curl \
    apt-transport-https \
    gnupg2 \
    unixodbc-dev \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl -fsSL https://packages.microsoft.com/config/debian/12/prod.list | tee /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# 4. Copiar e instalar las dependencias exactas de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el código del dashboard, la imagen y la carpeta de modelos
COPY app_final.py .
COPY logoexa.jpg .
COPY modelos_navales/ ./modelos_navales/

# 6. Exponer el puerto de Streamlit
EXPOSE 8501

# 7. Comando para ejecutar el Dashboard al levantar el contenedor
CMD ["streamlit", "run", "app_final.py", "--server.port=8501", "--server.address=0.0.0.0"]