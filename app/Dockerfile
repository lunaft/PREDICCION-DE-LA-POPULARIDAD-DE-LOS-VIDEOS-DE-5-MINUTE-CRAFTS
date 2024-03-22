# Usa la imagen base de Python
FROM python:3.9

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia el archivo requirements.txt al contenedor
COPY requirements.txt .

# Instala las dependencias del archivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia todos los archivos de la aplicación al contenedor
COPY . .

# Expone el puerto 5000 para que Flask pueda ser accesible desde fuera del contenedor
EXPOSE 5000

# Ejecuta la aplicación Flask cuando se inicie el contenedor
CMD ["python", "app.py"]
