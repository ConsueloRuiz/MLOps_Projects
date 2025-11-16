# Dockerfile

# Imagen base de Python ligera (mejora la portabilidad y seguridad)
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los requerimientos e instálalos
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia la aplicación (API) y los artefactos del modelo
COPY api/ /app/api/
COPY models/ /app/models/

# Puerto que expone la aplicación
EXPOSE 8000

# Comando para ejecutar la aplicación usando Uvicorn (servidor ASGI)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]