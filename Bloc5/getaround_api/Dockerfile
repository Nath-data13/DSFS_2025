# FROM python:3.11-slim

# WORKDIR /app

# # Copier et installer les dépendances
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# # Copier le reste des fichiers
# COPY . .

# # Exposer le port utilisé par Hugging Face pour les APIs
# EXPOSE 7860

# # Commande pour lancer l'API
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt ./ 
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers de ton projet
COPY . .

# Exposer le port utilisé par HF pour les APIs
EXPOSE 7860

# Lancer l'API avec uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--reload"]
