# Utiliza una imagen base de Python
FROM python:3.9

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia el archivo requirements.txt al contenedor
COPY requirements.txt .

# Instala las dependencias de la aplicación
RUN pip install -r requirements.txt

# Copia todo el contenido del directorio actual al contenedor
COPY . .

# Expone el puerto 5000 (o el puerto en el que se ejecute tu aplicación Flask)
EXPOSE 5000

# Comando para ejecutar tu aplicación Flask
CMD ["python", "app.py"]

# FROM python:3-alpine AS builder
 
# WORKDIR /app
 
# RUN python3 -m venv venv
# ENV VIRTUAL_ENV=/app/venv
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"
 
# COPY requirements.txt .
# RUN pip install -r requirements.txt
 
# # Stage 2
# FROM python:3-alpine AS runner
 
# WORKDIR /app
 
# COPY --from=builder /app/venv venv
# COPY app.py app.py
 
# ENV VIRTUAL_ENV=/app/venv
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# ENV FLASK_APP=app/app.py
 
# EXPOSE 8080
 
# CMD ["gunicorn", "--bind" , ":8080", "--workers", "2", "app:app"]