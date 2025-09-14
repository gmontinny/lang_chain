from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus

# Modelos de embedding disponíveis
AVAILABLE_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "multilingual-e5-large": "intfloat/multilingual-e5-large",
    "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
}

# Escolha o modelo a ser usado
SELECTED_MODEL = "all-MiniLM-L6-v2"

# Carregar o PDF
loader = PyPDFLoader("data/Edital Seletivo MTI SEPLAG 2025.pdf")
documents = loader.load()

# Dividir o texto em chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Modelo de embedding
model_name = AVAILABLE_MODELS[SELECTED_MODEL]
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Armazenar em um banco vetorial Milvus
vector_store = Milvus.from_documents(
    docs,
    embedding=embeddings,
    connection_args={"host": "127.0.0.1", "port": 19530},
)

print(f"Documentos armazenados no Milvus usando o modelo: {model_name}")

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
