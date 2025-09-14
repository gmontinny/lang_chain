# Extração de Informações de PDF com LangChain e Milvus

Este projeto demonstra como extrair informações de um arquivo PDF, gerar embeddings usando diferentes modelos de transformers (open source e pagos) e armazenar esses embeddings em um banco de dados vetorial Milvus. Em seguida, ele permite que você faça perguntas sobre o conteúdo do PDF e recupere as informações mais relevantes.

## Funcionalidades

-   **Processamento de PDF:** Carrega e divide o texto de arquivos PDF em pedaços gerenciáveis.
-   **Embeddings de Texto:** Usa as bibliotecas `sentence-transformers` e `langchain-openai` para gerar embeddings de texto.
-   **Suporte a Múltiplos Modelos:** Permite a fácil seleção entre vários modelos de embedding, tanto open source quanto pagos (OpenAI).
-   **Armazenamento Vetorial:** Integra-se com o Milvus para armazenar e pesquisar eficientemente os embeddings de texto.
-   **Recuperação de Informações:** Permite que os usuários façam perguntas em linguagem natural e recuperem as seções mais relevantes do PDF.

## Primeiros Passos

### Pré-requisitos

-   Python 3.8 ou superior
-   Docker (para executar o Milvus)
-   Uma chave de API da OpenAI (para usar os modelos da OpenAI)
-   Um arquivo PDF na pasta `data`

### Instalação

1.  **Clone o repositório:**
    ```sh
    git clone https://github.com/seu-usuario/seu-repositorio.git
    cd seu-repositorio
    ```

2.  **Instale as dependências:**
    ```sh
    pip install -r requirements.txt
    ```

### Executando o Milvus

Para iniciar uma instância do Milvus usando o Docker, execute o seguinte comando:

```sh
docker run -d --name milvus_bootcamp -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.2.11-standalone
```

## Uso

### Modelos Open Source (Hugging Face)

1.  **Adicione um PDF:** Coloque o arquivo PDF que você deseja processar no diretório `data`.

2.  **Selecione um Modelo:** Abra o arquivo `app.py` e altere a variável `SELECTED_MODEL` para escolher um dos modelos disponíveis:

    ```python
    # Escolha o modelo a ser usado
    SELECTED_MODEL = "all-MiniLM-L6-v2"  # Ou "all-mpnet-base-v2", "multilingual-e5-large", "bge-large-en-v1.5"
    ```

3.  **Execute o Script:**
    ```sh
    python app.py
    ```

### Modelos Pagos (OpenAI)

1.  **Configure a Chave de API:** Defina a sua chave de API da OpenAI como uma variável de ambiente. Substitua `sua-chave-de-api` pela sua chave real.

    No Linux/macOS:
    ```sh
    export OPENAI_API_KEY='sua-chave-de-api'
    ```

    No Windows (PowerShell):
    ```sh
    $env:OPENAI_API_KEY='sua-chave-de-api'
    ```

2.  **Execute o Script:**
    ```sh
    python app_openai.py
    ```

O script irá processar o PDF usando o modelo de embedding da OpenAI, armazenar os embeddings no Milvus e, em seguida, fazer uma pergunta de exemplo para recuperar e exibir os documentos relevantes.

## Modelos Disponíveis

### Open Source (em `app.py`)

-   `sentence-transformers/all-MiniLM-L6-v2`
-   `sentence-transformers/all-mpnet-base-v2`
-   `intfloat/multilingual-e5-large`
-   `BAAI/bge-large-en-v1.5`

### Pagos (em `app_openai.py`)

-   O modelo de embedding padrão da OpenAI (atualmente `text-embedding-ada-002`).

## Estrutura do Projeto

```
.
├── app.py              # Script para modelos de embedding open source
├── app_openai.py       # Script para modelos de embedding da OpenAI
├── requirements.txt    # As dependências do Python para o projeto
├── data/                 # O diretório onde você deve colocar seus arquivos PDF
│   └── ...
└── README.md           # Este arquivo
```
