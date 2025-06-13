✅ `README.md`

````markdown
🤖 AI FAQ Chatbot (LangChain + FastAPI)

This project is an end-to-end **AI-powered FAQ chatbot** that can answer domain-specific questions using a custom knowledge base. It uses:

- ✅ [FastAPI](https://fastapi.tiangolo.com/) for the backend
- ✅ [LangChain](https://www.langchain.com/) for chaining logic and question-answering
- ✅ [ChromaDB](https://www.trychroma.com/) as the vector store
- ✅ [Hugging Face Transformers](https://huggingface.co/) models for LLM and embeddings
- ✅ [JWT Auth](https://jwt.io/) for authentication
- ✅ Docker-ready setup (optional)
- ✅ Simple frontend with HTML/CSS/JS

---

🚀 Features

- 🔐 User authentication with JWT tokens
- 🧠 Vector-based search over your own data
- 💬 Secure chat interface powered by Hugging Face models
- ⚙️ Modular backend using FastAPI
- 📦 Easily deployable with Docker

---

🧩 Tech Stack

| Component     | Tool                          |
|---------------|-------------------------------|
| Backend       | FastAPI, Python 3.10+         |
| LLM & Embeds  | HuggingFace, LangChain        |
| Vector Store  | ChromaDB                      |
| Auth          | OAuth2 + JWT (Bearer Tokens)  |
| Frontend      | HTML + CSS + JavaScript       |
| Deployment    | Docker, Railway/Render/EC2    |

---

🛠️ Installation & Setup

 1. 📦 Clone the Repo

```bash
git clone https://github.com/yourusername/ai-faq-chatbot.git
cd ai-faq-chatbot
````

2. 🐍 Create & Activate Virtual Env

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. 📚 Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure to also install `chromadb` and `transformers`:

```bash
pip install chromadb transformers accelerate
```

If you're using M1/M2 Mac and get MPS warnings, update `torch`:

```bash
pip install --upgrade torch torchvision torchaudio
```

---

 4. ⚙️ Environment Variables

Edit `backend/app/config.py` and set your:

```python
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
```

---

5. 🧠 Prepare Vector DB

Before running the app, ensure you have ingested your documents into ChromaDB. If not, create an ingestion script and run it once.

```bash
python ingest.py  # Make sure it saves to `vectorstore/chroma/`
```

---

6. 🚦 Run Backend API

```bash
uvicorn backend.app.main:app --reload
```

This will start the FastAPI app at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

7. 💻 Open the Frontend

The root FastAPI path serves the HTML UI from `frontend/`:

```
📁 frontend/
 ├── index.html
 ├── style.css
 └── app.js
```

Just open [http://localhost:8000](http://localhost:8000) in your browser.

---

🔐 Authentication

Use the default credentials to log in:

* **Username:** `admin`
* **Password:** `admin123`

It will return a `Bearer Token` used in headers for accessing the `/chat` route.

---

🐳 Docker (Optional)

To run inside Docker:

```bash
docker build -t ai-faq-chatbot .
docker run -p 8000:8000 ai-faq-chatbot
```

---

📁 Project Structure

```
ai-faq-chatbot/
│
├── backend/
│   ├── app/
│   │   ├── main.py         # FastAPI app
│   │   ├── auth.py         # JWT Auth logic
│   │   ├── chat.py         # LangChain + Chroma logic
│   │   └── config.py       # Secrets and config
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── vectorstore/            # Chroma vector DB
├── Dockerfile              # Optional Docker support
├── requirements.txt
├── README.md
└── LICENSE
```

---
📜 License

This project is licensed under the **MIT License**. See [`LICENSE`](./LICENSE) for details.

---
 
🤝 Contributing

PRs welcome! If you find bugs or want to suggest improvements, open an issue or submit a pull request.

---

📬 Contact

Made with by [Jenish Shekhada](mailto:your-email@example.com)
