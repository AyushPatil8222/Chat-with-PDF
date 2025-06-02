# 🧾 DocuDiscuss: Chat with PDF

DocuDiscuss is an interactive Streamlit app that lets you upload PDF documents and chat with them using Google's Gemini Pro AI. Simply upload your PDFs, and ask questions in natural language — the app extracts relevant information from your documents and provides detailed answers.

---

## 🚀 Features

- 📄 Upload one or more PDF files to process their content
- 🔍 Extract and split PDF text into manageable chunks for better context handling
- 🤖 Use Gemini Pro (Google Generative AI) to answer questions based on your PDFs
- 💬 Chat interface powered by Streamlit for real-time interaction
- 📚 Stores vector embeddings locally with FAISS for efficient similarity search
- 🧠 Provides detailed, context-aware answers or informs if the answer is unavailable in the documents

---

## 🛠️ Technologies Used

- **Streamlit** – Easy-to-use web app framework for Python  
- **Google Generative AI (Gemini Pro)** – Advanced LLM for natural language understanding  
- **PyPDF2** – PDF text extraction  
- **LangChain** – For text splitting, chaining, and vector store integration  
- **FAISS** – Fast vector similarity search library  
- **Python-dotenv** – Manage environment variables securely

