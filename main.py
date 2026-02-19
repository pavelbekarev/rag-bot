import streamlit as st
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import chromadb

# -----------------------------
# 1. Очистка и chunking
# -----------------------------
def clean_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text

def split_text(text, chunk_size=50, overlap=10):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# -----------------------------
# 2. Заголовок приложения
# -----------------------------
st.title("RAG Бот: поиск страниц PDF (локально)")

# -----------------------------
# 3. Загрузка PDF
# -----------------------------
uploaded_file = st.file_uploader("Загрузите PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    documents = []
    metadatas = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            text = clean_text(text)
            chunks = split_text(text)
            for chunk in chunks:
                documents.append(chunk)
                metadatas.append({"page": page_number + 1, "text_preview": chunk[:100]})

    st.write(f"PDF загружен, всего chunks: {len(documents)}")

    # -----------------------------
    # 4. Модель для embeddings
    # -----------------------------
    st.write("Загрузка модели embeddings...")
    embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")

    # -----------------------------
    # 5. ChromaDB
    # -----------------------------
    client = chromadb.Client()
    collection = client.get_or_create_collection("lecture_collection")

    embeddings = embedding_model.encode(["passage: " + d for d in documents]).tolist()
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=[str(i) for i in range(len(documents))]
    )

    st.write("База готова!")

    # -----------------------------
    # 6. Вопрос пользователя
    # -----------------------------
    question = st.text_input("Введите вопрос:")

    if question:
        query_embedding = embedding_model.encode("query: " + question).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=5)

        # -----------------------------
        # 7. Красивый вывод страниц и превью текста
        # -----------------------------
        st.subheader("Релевантные страницы PDF:")
        for idx, meta in enumerate(results["metadatas"][0]):
            page_num = meta["page"]
            preview = meta["text_preview"]
            st.markdown(f"**Страница {page_num}** — _{preview}…_")
