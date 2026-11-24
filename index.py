# =====================================================================
# EJERCICIO 1: crear un índice serverless, cargar vectores y ver estadísticas
# =====================================================================
# Antes de correrlo (una sola vez), en la terminal:
#     pip install pinecone
# y guardá tu clave en la variable de entorno PINECONE_API_KEY.
# =====================================================================
import os
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# El namespace (partición) donde vamos a guardar los vectores de prueba
NAMESPACE = "dev-environment"


async def setup_vector_infrastructure(index_name: str, dimension: int):
    """Configura Pinecone Serverless y hace una carga inicial."""

    # 1. Inicializar el cliente: es la "conexión" con tu cuenta de Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # 2. Crear el índice SOLO si no existe.
    #    list_indexes().names() devuelve los nombres de tus índices.
    #    Así el programa se puede correr muchas veces sin dar error.
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,   # debe coincidir con el modelo de embeddings
            metric="cosine",       # la métrica habitual para texto
            # ServerlessSpec: en qué nube y región vive el índice
            spec=ServerlessSpec(cloud="aws", region="us-east-1"))

        # Crear un índice tarda unos segundos. Esperamos a que esté listo
        # antes de usarlo; si no, el upsert puede fallar.
        while not pc.describe_index(index_name).status["ready"]:
            await asyncio.sleep(1)

    # 3. Conectarse al índice: "index" es el objeto con el que operamos
    index = pc.Index(index_name)

    # 4. Upsert de prueba dentro de un namespace.
    #    Cada vector es una tupla (id, valores, metadata).
    #    [0.1] * dimension crea una lista de 1536 números: es un vector
    #    "de mentira", solo para probar la infraestructura.
    vectors = [
        ("vec1", [0.1] * dimension, {"topic": "infraestructura", "priority": "high"}),
        ("vec2", [0.2] * dimension, {"topic": "seguridad", "priority": "medium"}),
        ("vec3", [0.3] * dimension, {"topic": "costos", "priority": "low"}),
    ]
    #    upsert = update + insert: si el id existe lo actualiza; si no, lo crea.
    #    Por eso, al correr el programa otra vez, no se duplican.
    index.upsert(vectors=vectors, namespace=NAMESPACE)

    #    Las estadísticas tardan unos segundos en reflejar la carga
    await asyncio.sleep(5)

    # 5. Devolver e imprimir las estadísticas del índice
    stats = index.describe_index_stats()
    print(stats)
    return stats


if __name__ == "__main__":
    # Dimensión 1536: la de text-embedding-3-small de OpenAI
    asyncio.run(setup_vector_infrastructure("mi-indice-serverless", 1536))