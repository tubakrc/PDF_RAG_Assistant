"""
Day06: İleri RAG – Streamlit PDF Soru-Cevap Uygulaması
-------------------------------------------------------
Pipeline (ders notlarındaki 3 adım):
  1. İndeksle  : PDF -> Document (metadata ile) -> RecursiveCharacterTextSplitter -> Embedding -> PGVector
  2. Yakala    : Basic / Rewrite-Retrieve / Multi Query / RAG-Fusion / HyDE
  3. Üret      : Sadece bağlama dayalı cevap + kaynaklar (dosya, sayfa)
"""

import hashlib
import io
import os

import pypdf
import streamlit as st
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="PDF RAG Asistanı", page_icon="📚", layout="wide")

TECHNIQUES = {
    "Basic": "Soruyu doğrudan vektör DB'de ara.",
    "Rewrite-Retrieve": "LLM soruyu arama için yeniden yazar, sonra aranır.",
    "Multi Query": "LLM sorunun birkaç farklı versiyonunu üretir, sonuçlar birleştirilir.",
    "RAG-Fusion": "Multi Query + sonuçlar Reciprocal Rank Fusion ile yeniden sıralanır.",
    "HyDE (Hypothetical Document)": "LLM varsayımsal bir cevap yazar, bu metinle arama yapılır.",
}

SYSTEM_PROMPT = """Answer the question based only on the following context:
{context}

If the answer is not in the context, say you could not find it in the documents.
Answer in the same language as the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)

st.title("📚 PDF RAG Asistanı")
st.caption("PDF'lerini yükle, indeksle ve sorularını sor. Cevaplar sadece dokümanlara dayanır.")

# --------------------------------------------------------------------------- #
# Sidebar: ayarlar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Ayarlar")

    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
    )
    connection = st.text_input(
        "PostgreSQL bağlantısı",
        value="postgresql+psycopg://ai:ai@localhost:6024/ai_db",
    )
    collection_name = st.text_input("Collection adı", value="rag-pdf-2-cohort")

    st.subheader("Modeller")
    llm_model = st.text_input("LLM modeli", value="gpt-5.6-luna")
    embedding_model = st.text_input("Embedding modeli", value="text-embedding-3-small")

    st.subheader("Chunking")
    chunk_size = st.slider(
        "Chunk size", 300, 2000, 800, 50,
        help="Kısa metinler: 300-600 | Makale/ders notu: 600-1000 | Uzun metinler: 1000-2000",
    )
    overlap_pct = st.slider(
        "Chunk overlap (% of chunk size)", 0, 30, 15, 1,
        help="Önerilen: chunk size'ın %10-20'si",
    )
    chunk_overlap = int(chunk_size * overlap_pct / 100)
    st.caption(f"Overlap: **{chunk_overlap}** karakter")
    if not 10 <= overlap_pct <= 20:
        st.warning("Önerilen overlap aralığı %10-20.")

    st.subheader("Retrieval")
    technique = st.selectbox("RAG query tekniği", list(TECHNIQUES))
    st.caption(TECHNIQUES[technique])
    top_k = st.slider("Top-K", 1, 10, 4)
    n_queries = st.slider(
        "Üretilecek sorgu sayısı", 2, 6, 3,
        disabled=technique not in ("Multi Query", "RAG-Fusion"),
    )

if not api_key:
    st.info("Devam etmek için sol menüden OpenAI API key gir.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key


# --------------------------------------------------------------------------- #
# Kaynaklar (cache'li)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_db(connection: str, collection_name: str, embedding_model: str, _key: str):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,  # metadata'ları kolay aranabilir şekilde sakla
    )


@st.cache_resource(show_spinner=False)
def get_llm(model: str, _key: str):
    return ChatOpenAI(model_name=model, temperature=0)


def check_connection(connection: str, timeout: int = 5) -> str | None:
    """Veritabanına kısa zaman aşımıyla bağlanmayı dener. Hata varsa mesajını döner."""
    from sqlalchemy import create_engine, text

    try:
        engine = create_engine(connection, connect_args={"connect_timeout": timeout})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return None
    except Exception as e:
        return str(e).splitlines()[0]


with st.spinner("⏳ PostgreSQL veritabanına bağlanılıyor... (en fazla 5 sn)"):
    conn_error = check_connection(connection)

if conn_error:
    st.error("❌ PostgreSQL'e bağlanılamadı.")
    st.markdown(
        "- Docker'daki pgvector container'ı çalışıyor mu? → `docker compose up -d`\n"
        "- Bağlantı adresindeki **port** (varsayılan `6024`), kullanıcı ve şifre doğru mu?\n"
        "- Docker Desktop açık mı?"
    )
    st.code(conn_error)
    st.stop()

with st.spinner("⏳ Vektör veritabanı hazırlanıyor (ilk açılışta tablolar ve pgvector eklentisi kurulur)..."):
    try:
        db = get_db(connection, collection_name, embedding_model, api_key)
    except Exception as e:
        st.error(f"Vektör veritabanı hazırlanamadı: {e}")
        st.stop()

llm = get_llm(llm_model, api_key)


# --------------------------------------------------------------------------- #
# 1. Adım: İndeksleme
# --------------------------------------------------------------------------- #
def pdf_to_documents(file_name: str, data: bytes) -> list[Document]:
    """Her sayfa bir Document; metadata: source + page."""
    reader = pypdf.PdfReader(io.BytesIO(data))
    docs = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": file_name, "page": page_number},
                )
            )
    return docs


def chunk_id(doc: Document) -> str:
    """Aynı chunk tekrar yüklenirse çoğalmasın (upsert)."""
    raw = f"{doc.metadata['source']}|{doc.metadata['page']}|{doc.metadata['start_index']}|{doc.page_content}"
    return hashlib.sha1(raw.encode()).hexdigest()


def index_pdfs(files) -> tuple[int, int]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    total_pages, total_chunks = 0, 0
    for f in files:
        raw_docs = pdf_to_documents(f.name, f.getvalue())
        chunks = splitter.split_documents(raw_docs)
        ids = [chunk_id(c) for c in chunks]
        for i in range(0, len(chunks), 100):  # batch'le
            db.add_documents(chunks[i : i + 100], ids=ids[i : i + 100])
        total_pages += len(raw_docs)
        total_chunks += len(chunks)
    return total_pages, total_chunks


def list_indexed_sources() -> list[str]:
    """Collection'daki benzersiz dosya adları (metadata.source)."""
    try:
        with db._make_sync_session() as session:
            from sqlalchemy import text

            rows = session.execute(
                text(
                    "SELECT DISTINCT e.cmetadata->>'source' "
                    "FROM langchain_pg_embedding e "
                    "JOIN langchain_pg_collection c ON c.uuid = e.collection_id "
                    "WHERE c.name = :name ORDER BY 1"
                ),
                {"name": collection_name},
            ).fetchall()
        return [r[0] for r in rows if r[0]]
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# 2. Adım: Retrieval teknikleri
# --------------------------------------------------------------------------- #
def _search(query: str, k: int, flt: dict | None) -> list[Document]:
    return db.similarity_search(query, k=k, filter=flt)


def _doc_key(doc: Document) -> tuple:
    return (
        doc.metadata.get("source"),
        doc.metadata.get("page"),
        doc.metadata.get("start_index"),
    )


def _llm_text(system: str, user: str) -> str:
    chain = ChatPromptTemplate.from_messages([("system", system), ("human", "{q}")]) | llm | StrOutputParser()
    return chain.invoke({"q": user}).strip()


def rewrite_query(question: str) -> str:
    return _llm_text(
        "Rewrite the user's question into a clear, specific search query for a "
        "vector database of documents. Keep the original language. "
        "Return only the rewritten query.",
        question,
    )


def generate_queries(question: str, n: int) -> list[str]:
    out = _llm_text(
        f"Generate {n} different versions of the user's question to retrieve relevant "
        "documents from a vector database. Use different wording and perspectives. "
        "Keep the original language. Return one question per line, no numbering.",
        question,
    )
    queries = [q.strip("-•0123456789. ").strip() for q in out.splitlines() if q.strip()]
    return queries[:n]


def generate_hypothetical_doc(question: str) -> str:
    return _llm_text(
        "Write a short, factual passage (about 100 words) that would answer the "
        "user's question, as if it came from a document. Keep the original language.",
        question,
    )


def reciprocal_rank_fusion(result_lists: list[list[Document]], k: int = 60) -> list[Document]:
    scores: dict[tuple, float] = {}
    docs: dict[tuple, Document] = {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            key = _doc_key(doc)
            docs[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked]


def retrieve(question: str, technique: str, k: int, flt: dict | None, n: int):
    """Döner: (docs, debug_bilgisi)"""
    if technique == "Basic":
        return _search(question, k, flt), {}

    if technique == "Rewrite-Retrieve":
        rewritten = rewrite_query(question)
        return _search(rewritten, k, flt), {"Yeniden yazılan sorgu": rewritten}

    if technique in ("Multi Query", "RAG-Fusion"):
        queries = [question] + generate_queries(question, n)
        result_lists = [_search(q, k, flt) for q in queries]
        if technique == "RAG-Fusion":
            docs = reciprocal_rank_fusion(result_lists)[:k]
        else:  # Multi Query: benzersiz birleşim
            seen, docs = set(), []
            for results in result_lists:
                for d in results:
                    if _doc_key(d) not in seen:
                        seen.add(_doc_key(d))
                        docs.append(d)
        return docs, {"Üretilen sorgular": queries}

    if technique.startswith("HyDE"):
        hypo = generate_hypothetical_doc(question)
        return _search(hypo, k, flt), {"Varsayımsal doküman": hypo}

    raise ValueError(technique)


# --------------------------------------------------------------------------- #
# 3. Adım: Generation
# --------------------------------------------------------------------------- #
def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('source')} – sayfa {d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    )


def stream_answer(question: str, docs: list[Document]):
    chain = answer_prompt | llm | StrOutputParser()
    yield from chain.stream({"context": format_context(docs), "question": question})


def render_sources(docs: list[Document], debug: dict):
    with st.expander(f"📎 Kaynaklar ({len(docs)})"):
        for key, value in debug.items():
            st.markdown(f"**{key}**")
            if isinstance(value, list):
                for v in value:
                    st.markdown(f"- {v}")
            else:
                st.write(value)
            st.divider()
        for i, d in enumerate(docs, start=1):
            st.markdown(f"**Kaynak {i}** — `{d.metadata.get('source')}`, sayfa {d.metadata.get('page')}")
            st.caption(d.page_content)


# --------------------------------------------------------------------------- #
# Arayüz
# --------------------------------------------------------------------------- #
tab_chat, tab_docs = st.tabs(["💬 Sohbet", "📄 Dokümanlar"])
sources = list_indexed_sources()

with tab_docs:
    uploaded = st.file_uploader("PDF dosyaları (birden fazla seçebilirsin)", type="pdf", accept_multiple_files=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📥 İndeksle", type="primary", disabled=not uploaded):
            with st.spinner("⏳ PDF'ler okunuyor, parçalanıyor ve embedding'leniyor (OpenAI API'ye istek atılır, dosya boyutuna göre birkaç dakika sürebilir)..."):
                pages, chunks = index_pdfs(uploaded)
            st.success(f"{len(uploaded)} dosya, {pages} sayfa, {chunks} chunk indekslendi.")
    with col2:
        if st.button("🗑️ Collection'ı temizle"):
            db.delete_collection()
            get_db.clear()
            st.session_state.pop("messages", None)
            st.success("Collection silindi. Sayfayı yenile.")
            st.stop()

    st.subheader("İndekslenmiş dosyalar")
    if sources:
        for s in sources:
            st.markdown(f"- {s}")
    else:
        st.caption("Henüz doküman yok.")

with tab_chat:
    selected = st.multiselect(
        "Aranacak dosyalar (boş = hepsi)", sources, help="Metadata filtresi ile aramayı daraltır."
    )
    flt = {"source": {"$in": selected}} if selected else None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("docs"):
                render_sources(m["docs"], m.get("debug", {}))

    if question := st.chat_input("Dokümanlar hakkında bir soru sor..."):
        if not sources:
            st.warning("Önce 'Dokümanlar' sekmesinden PDF yükleyip indeksle.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner(f"{technique} ile ilgili parçalar aranıyor..."):
                docs, debug = retrieve(question, technique, top_k, flt, n_queries)
            answer = st.write_stream(stream_answer(question, docs))
            render_sources(docs, debug)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "docs": docs, "debug": debug}
        )
