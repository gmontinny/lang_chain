import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Milvus

# Verifique se a chave da API da OpenAI está configurada
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("A variável de ambiente OPENAI_API_KEY não foi configurada.")

# Carregar o PDF
loader = PyPDFLoader("data/Edital Seletivo MTI SEPLAG 2025.pdf")
documents = loader.load()

# Dividir o texto em chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Modelo de embedding da OpenAI
embeddings = OpenAIEmbeddings()

# Armazenar em um banco vetorial Milvus
vector_store = Milvus.from_documents(
    docs,
    embedding=embeddings,
    connection_args={"host": "127.0.0.1", "port": 19530},
    collection_name="openai_collection"  # Usar uma coleção diferente para evitar conflitos
)

print("Documentos armazenados no Milvus usando o modelo da OpenAI.")

# Criar um retriever
retriever = vector_store.as_retriever()

# Fazer uma pergunta
question = "Quais são os requisitos para o cargo de Analista de Sistemas?"
docs_retrieved = retriever.get_relevant_documents(question)

# Imprimir os resultados
print(f"\nPergunta: {question}")
print("Documentos relevantes encontrados:")
for doc in docs_retrieved:
    print(doc.page_content)
