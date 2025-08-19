# End-to-End Document Q&A RAG App

### Overview

This project is an end-to-end Document Q&A application built with a Retrieval-Augmented Generation (RAG) architecture. It allows users to upload a document, and then ask questions about its content. The application leverages the power of the Gemma large language model and the high-speed inference of the Groq API to provide quick and accurate answers.

### Features

* **Document Processing**: Ingests and processes a variety of document types.
* **Retrieval-Augmented Generation (RAG)**: Retrieves relevant text snippets from the document to provide context for the LLM.
* **Fast Inference**: Utilizes the Groq API for incredibly fast and low-latency responses.
* **Gemma Integration**: Uses the Gemma model for high-quality, human-like text generation.
* **Intuitive Interface**: A simple and easy-to-use interface for uploading documents and asking questions.

### Technologies Used

* **Python**: The core programming language for the backend logic.
* **Groq API**: For high-speed, on-demand inference of LLMs.
* **Gemma**: The large language model used for generation.
* **LangChain**: For document loading, chunking, and vector storage.
* **FAISS**: For efficient vector search and retrieval.
* **Streamlit**: For building the web application interface.

### Setup

#### Prerequisites

Make sure you have the following installed on your system:

* Python 3.8+
* Git

#### Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/BLVCK-MAMBA-6/Gemma-Groq-RAG-App.git](https://github.com/BLVCK-MAMBA-6/Gemma-Groq-RAG-App.git)
    cd Gemma-Groq-RAG-App
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

#### API Key Configuration

You will need an API key from Groq to use this application.

1.  Sign up or log in to the [Groq Console](https://console.groq.com/).
2.  Generate a new API key.
3.  Create a file named `.env` in the root directory of your project.
4.  Add your API key to this file as shown below:
    ```
    GROQ_API_KEY="your_api_key_here"
    ```

### How to Run the App

Once you have completed the setup, run the application using the appropriate command (e.g., if you are using Streamlit):

```bash
streamlit run app.py
