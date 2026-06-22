# Company-knowledge-rag-chatbot
An end-to-end RAG chatbot that answers company-related questions using Gemini, LangChain, BGE-M3 embeddings, ChromaDB and Gradio.

# Company Knowledge RAG Chatbot

A Retrieval-Augmented Generation chatbot that answers questions using information stored in a company knowledge base.

The application loads company documents, divides them into smaller chunks, converts those chunks into embedding vectors, stores them in ChromaDB, retrieves relevant information for a user's question and sends the retrieved context to Google Gemini for generating a grounded response.

## Project Overview

Large Language Models may provide incorrect or outdated answers when they do not have access to private company information.

This project solves that problem using Retrieval-Augmented Generation, also known as RAG.

Instead of asking the language model to answer using only its training data, the application first searches a custom company knowledge base and provides the most relevant information to the model.

The chatbot can answer questions related to:

- Company information
- Employees
- Products
- Contracts
- Internal company documents

## RAG Workflow

```text
Company Documents
        ↓
Document Loading
        ↓
Text Chunking
        ↓
BGE-M3 Embeddings
        ↓
ChromaDB Vector Store
        ↓
Semantic Retrieval
        ↓
Context Augmentation
        ↓
Gemini Response Generation
        ↓
Gradio Chat Interface
```

## Features

- Loads multiple Markdown documents from different folders
- Adds document-category metadata
- Splits documents using recursive text chunking
- Generates embeddings using `BAAI/bge-m3`
- Stores embeddings in a persistent ChromaDB vector database
- Retrieves semantically relevant document chunks
- Generates grounded answers using Google Gemini
- Provides a browser-based chatbot using Gradio
- Visualizes vector embeddings in two and three dimensions using t-SNE and Plotly
- Returns a fallback response when information is unavailable

## Technologies Used

- Python
- LangChain
- Google Gemini
- Hugging Face Embeddings
- BAAI BGE-M3
- ChromaDB
- Gradio
- Plotly
- Scikit-learn
- tiktoken
- NumPy

## Project Structure

```text
company-knowledge-rag-chatbot/
│
├── app.py
├── ingest.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── data/
│   ├── company/
│   ├── employees/
│   ├── products/
│   └── contracts/
│
└── vector_db/
```

## How It Works

### 1. Document Loading

The application loads Markdown documents from the knowledge-base directory using LangChain's `DirectoryLoader`.

Each document receives metadata representing its category, such as:

```text
company
employees
products
contracts
```

### 2. Text Chunking

Large documents are divided into smaller overlapping sections using `RecursiveCharacterTextSplitter`.

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=200
)
```

The overlap helps preserve context between consecutive chunks.

### 3. Embedding Generation

Each chunk is converted into a numerical embedding vector using:

```text
BAAI/bge-m3
```

BGE-M3 is a multilingual embedding model suitable for semantic retrieval.

### 4. Vector Storage

The embeddings and document metadata are stored in ChromaDB.

ChromaDB allows the application to search for chunks that are semantically similar to the user's question.

### 5. Information Retrieval

When a user submits a question, the retriever converts the question into an embedding and searches ChromaDB for the most relevant chunks.

### 6. Response Generation

The retrieved chunks are added to the system prompt and sent to Gemini.

Gemini then generates an answer grounded in the company documents instead of depending only on its general training knowledge.

### 7. User Interface

A Gradio chat interface allows users to interact with the RAG system through a web browser.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/company-knowledge-rag-chatbot.git
cd company-knowledge-rag-chatbot
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Never upload your real `.env` file or API key to GitHub.

## Running the Application

First, place the company Markdown files inside the appropriate folders under `data/`.

Then create the vector database:

```bash
python ingest.py
```

Start the chatbot:

```bash
python app.py
```

Gradio will display a local URL similar to:

```text
http://127.0.0.1:7860
```

Open the URL in your browser to use the chatbot.

## Example Questions

```text
Who is Averi Lancaster?
```

```text
What products does the company offer?
```

```text
Who works in the company?
```

```text
What information is available about the company's contracts?
```

## Vector Visualization

The project uses t-SNE to reduce high-dimensional embedding vectors into two and three dimensions.

Plotly is then used to display interactive visualizations of document embeddings.

These visualizations help demonstrate how semantically related document chunks form groups inside the vector space.

## Example Retrieval Code

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

docs = retriever.invoke(question)

context = "\n\n".join(
    document.page_content
    for document in docs
)
```

## Example Response Generation

```python
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    temperature=0,
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": question
        }
    ]
)
```

## Current Limitations

- It currently uses basic similarity retrieval
- Conversation history is not yet included in retrieval
- Retrieved answers do not display formal citations in the user interface
- There is no reranking stage
- The system has not yet been evaluated using a dedicated RAG evaluation framework
- Knowledge-base updates currently require re-ingestion

## Future Improvements

- Add hybrid search using semantic and keyword retrieval
- Add a reranker for better document selection
- Display document names and citations with every answer
- Add conversational memory
- Add PDF, DOCX and webpage support
- Add metadata filtering
- Add RAG evaluation using RAGAS or DeepEval
- Add Docker support
- Add automated tests
- Deploy the chatbot to Hugging Face Spaces, Render or a cloud platform
- Create an admin interface for uploading new documents

## Learning Outcomes

This project helped me understand:

- Retrieval-Augmented Generation architecture
- Document ingestion pipelines
- Chunking strategies
- Embedding models
- Vector databases
- Semantic similarity search
- Prompt grounding
- LangChain retrievers
- Gemini API integration
- Gradio application development
- Vector visualization

## Disclaimer

This project is created for educational and portfolio purposes.

The chatbot should only answer using information available in the provided knowledge base. Its output should be verified before being used in business-critical situations.

## Author

**Mohit Sharma**

- GitHub: `https://github.com/your-username`
- Email: `your-email@example.com`

## License

This project is licensed under the MIT License.
