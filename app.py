import os
import json
from datetime import datetime
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
import shutil
# ─── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
os.makedirs("uploads", exist_ok=True)
os.makedirs("vector_store", exist_ok=True)
if not os.path.exists("chat_logs.txt"):
    open("chat_logs.txt", "w", encoding="utf-8").close()

METADATA_PATH = os.path.join("uploads", "metadata.json")
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
else:
    metadata = {}
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

# ─── Helpers ────────────────────────────────────────────────────────────────────

def save_metadata():
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def load_file_content(file_path):
    ext = file_path.lower().split('.')[-1]
    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "docx":
        loader = Docx2txtLoader(file_path)
    elif ext == "pptx":
        loader = UnstructuredPowerPointLoader(file_path)
    else:
        return None
    return loader.load()

def index_all_files():
    file_paths = [
        os.path.join("uploads", fn)
        for fn in os.listdir("uploads")
        if fn.lower().endswith((".pdf", ".docx", ".pptx"))
    ]
    all_docs = []
    for path in file_paths:
        docs = load_file_content(path)
        if docs:
            all_docs.extend(docs)
    
    # ── NEW GUARD ──────────────────────────────────────────────────────────────
    if not all_docs:
        # remove any old index so your app knows there's nothing to search
        if os.path.exists("vector_store"):
            shutil.rmtree("vector_store")
        return
    # ── END GUARD ──────────────────────────────────────────────────────────────

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(all_docs, embeddings)
    vectordb.save_local("vector_store")

def is_arabic(text: str) -> bool:
    """Return True if the text contains any Arabic character."""
    return any('\u0600' <= ch <= '\u06FF' for ch in text)

def get_pdf_response(query: str) -> str:
    """إرجاع رد على سؤال المستخدم استنادًا إلى الملفات المفهرسة."""
    # ─── New: Immediately bail if no uploaded docs exist ───────────────────────
    uploaded = [
        fn for fn in os.listdir("uploads")
        if fn.lower().endswith((".pdf", ".docx", ".pptx"))
    ]
    if not uploaded:
        return "المعلومة غير متوفرة."

    try:
        vectordb = FAISS.load_local(
            "vector_store",
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True
        )
        huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not huggingface_api_token:
            return "لم يتم إعداد رمز API الخاص بـ Hugging Face."

        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            huggingfacehub_api_token=huggingface_api_token,
            task="text-generation",
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(),
            chain_type="stuff",
        )

        # If the user question is in Arabic, prefix to enforce Arabic response
        if is_arabic(query):
            prompt = "أجب عن السؤال التالي باللغة العربية فقط وبوضوح:\n" + query
        else:
            prompt = query

        ans = qa.run(prompt).strip()
        return ans or "المعلومة غير متوفرة."
    except Exception as e:
        return f"حدث خطأ: {str(e)}"

def log_user_query(q: str, a: str):
    """تسجيل استعلام المستخدم والرد في ملف السجل."""
    with open("chat_logs.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.utcnow().isoformat()}] Q: {q}\nA: {a}\n\n")

def display_chat_logs():
    with open("chat_logs.txt", "r", encoding="utf-8") as f:
        logs = f.read()
    st.text_area("All user Q&A logs", logs, height=400)

# ─── Session State Init ──────────────────────────────────────────────────────────

if "mode" not in st.session_state:
    st.session_state.mode = "user"
if "login_pending" not in st.session_state:
    st.session_state.login_pending = False

# ─── Interfaces ──────────────────────────────────────────────────────────────────

def show_user_interface():
    st.title("🤖 روبوت دردشة مدعوم بالملفات")
    q = st.text_input("اكتب سؤالك هنا:")
    if st.button("إرسال") and q:
        a = get_pdf_response(q)
        st.markdown(f"البوت: {a}")
        log_user_query(q, a)

def admin_login():
    with st.form("login", clear_on_submit=True):
        name = st.text_input("اسم المدير")
        admin_id = st.text_input("رمز الدخول", type="password")
        ok = st.form_submit_button("دخول")
        if ok:
            if name.lower() == "admin" and admin_id == "1100":
                st.session_state.mode = "admin"
                st.session_state.login_pending = False
                st.rerun()
            else:
                st.error("❌ بيانات الدخول غير صحيحة")

def manage_pdf_files():
    st.markdown("### رفع ملف جديد")
    up = st.file_uploader("اختر ملف (PDF, DOCX, PPTX)", type=["pdf", "docx", "pptx"])
    if up:
        ext = up.name.split(".")[-1].lower()
        if ext not in ["pdf", "docx", "pptx"]:
            st.warning("❌ نوع الملف غير مدعوم.")
        else:
            path = os.path.join("uploads", up.name)
            with open(path, "wb") as f:
                f.write(up.getbuffer())
            metadata[up.name] = {
                "uploader": "admin",
                "upload_date": datetime.utcnow().isoformat(),
                "file_type": up.type or "unknown",
                "file_size_kb": round(len(up.getvalue()) / 1024, 2)
            }
            save_metadata()
            index_all_files()
            st.success(f"تم رفع ومعالجة {up.name}")

    st.markdown("### قاعدة المعرفة")
    sort_by = st.selectbox("ترتيب حسب", ["الاسم ⬆", "ال Naam ⬇", "التاريخ ⬆", "التاريخ ⬇"])
    items = list(metadata.items())
    if "الاسم" in sort_by:
        items.sort(key=lambda x: x[0], reverse="⬇" in sort_by)
    else:
        items.sort(key=lambda x: x[1].get("upload_date", ""), reverse="⬇" in sort_by)

    for fn, meta in items:
        st.markdown(
            f"📄 **{fn}** "
            f"({meta.get('file_type', 'نوع غير معروف')}, "
            f"{meta.get('file_size_kb', '?')} KB) - "
            f"بتاريخ {meta.get('upload_date', 'تاريخ غير معروف')}, "
            f"رفعه: {meta.get('uploader', 'غير معروف')}"
        )
        c1, c2, c3 = st.columns([3, 1, 1])
        if c2.button("✏ تعديل", key=f"edit_{fn}"):
            new_upl = c1.text_input("اسم الرافع:", meta.get("uploader", ""), key=f"u_in_{fn}")
            save = c3.button("حفظ", key=f"save_{fn}")
            if save:
                metadata[fn]["uploader"] = new_upl
                save_metadata()
                st.success("تم تحديث البيانات.")
        if c3.button("🗑 حذف", key=f"del_{fn}"):
            os.remove(os.path.join("uploads", fn))
            metadata.pop(fn, None)
            save_metadata()
            index_all_files()
            st.warning(f"تم حذف {fn}")

def show_admin_interface():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.header("📜 سجل المستخدمين")
        display_chat_logs()
    with col2:
        st.header("💬 دردشة اختبارية")
        q = st.text_input("اكتب سؤالك كمدير:")
        if st.button("اسأل البوت") and q:
            a = get_pdf_response(q)
            st.markdown(f"البوت: {a}")
    with col3:
        st.header("📂 إدارة الملفات")
        manage_pdf_files()

def show_settings():
    st.title("⚙️ الإعدادات")
    st.text_input("اسم المستخدم:")
    st.text_input("البريد الإلكتروني:")
    st.text_input("كلمة المرور:", type="password")
    st.button("حفظ التغييرات")

# ─── Main UI ─────────────────────────────────────────────────────────────────────

st.sidebar.title("الوضع الحالي")
mode_choice = st.sidebar.radio("انتقل إلى:", ["الدردشة", "الإعدادات"])

if mode_choice == "الإعدادات":
    show_settings()
else:
    if st.session_state.mode == "user":
        if st.sidebar.button("الوضع الإداري"):
            st.session_state.login_pending = True
            st.rerun()
        if st.session_state.login_pending:
            admin_login()
        else:
            show_user_interface()
    else:
        if st.sidebar.button("الوضع العادي"):
            st.session_state.mode = "user"
            st.session_state.login_pending = False
            st.rerun()
        show_admin_interface()
