# 1. Crear el entorno virtual (una sola vez)
python -m venv venv

# 2. Activarlo (en Git Bash es Scripts, no bin)
source venv/Scripts/activate

# 3. Instalar la librería (una sola vez)
pip install pinecone python-dotenv

# 5. Correr el script
python index.py
