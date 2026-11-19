"""M&A Playbook RAG assistant (LangChain + FAISS). Answers ONLY from M_A_Due_Diligence_Playbook.pdf.
Usage:  python rag_app.py "What are earn-outs?"      or      python rag_app.py   (interactive)
"""
import sys
import os, re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PDF_PATH = "M_A_Due_Diligence_Playbook.pdf"   # the ONLY knowledge source
INDEX_DIR = "faiss_index"

def clean_text(t: str) -> str:
    """The PDF exports a few odd glyphs: bullets as \x7f, non-breaking hyphens as a box, 'M&A;' for 'M&A'."""
    t = t.replace("\x7f", "- ").replace("\u25a0", "-").replace("M&A;", "M&A").replace("R&W;", "R&W")
    return re.sub(r"[ \t]+", " ", t)

def load_pages(path=PDF_PATH):
    pages = PyPDFLoader(path).load()          # one Document per page, metadata has page (0-indexed)
    for p in pages:
        p.page_content = clean_text(p.page_content)
    return pages

def make_chunks(pages, chunk_size=400, chunk_overlap=60):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "; ", ". ", " ", ""],   # prefer to break at bullets/lines first
        add_start_index=True)
    return splitter.split_documents(pages)

def get_embeddings():
    """OpenAI embeddings if a key is set (the playbook itself mentions FAISS+OpenAI), otherwise a free local model."""
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vectorstore(chunks, embeddings, save_to=INDEX_DIR):
    vs = FAISS.from_documents(chunks, embeddings)   # only chunks from the playbook go in
    if save_to: vs.save_local(save_to)
    return vs

NOT_FOUND = "I couldn't find that in the M&A Playbook."

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful assistant that answers questions about the M&A Due Diligence Playbook.\n"
     "Rules:\n"
     "1. Use ONLY the context provided below. Do not use outside knowledge, even if you know the answer.\n"
     "2. If the context does not contain the answer, reply exactly: \"" + NOT_FOUND + "\" "
     "and, if helpful, mention what the playbook does cover on a nearby topic.\n"
     "3. Never guess or invent figures, clauses, names or case-study details.\n"
     "4. Keep the answer concise but complete, and cite the page like [p.1].\n\n"
     "Context:\n{context}"),
    ("human", "{question}")])

def get_llm():
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)   # temperature 0 = stick to the facts
    return None

def format_context(docs):
    return "\n\n".join(f"[p.{d.metadata.get('page', 0) + 1}] {d.page_content}" for d in docs)

def ask(question, vectorstore, llm=None, k=4):
    """question -> retrieve top-k chunks -> prompt -> LLM -> answer + the sources used."""
    hits = vectorstore.similarity_search_with_score(question, k=k)   # (Document, L2 distance) - lower = closer
    docs = [d for d, _ in hits]
    if llm is None:
        answer = "(Retrieval-only mode: set OPENAI_API_KEY to also generate an answer. The best-matching passages are shown below.)"
    else:
        chain = RAG_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": format_context(docs), "question": question})
    return {"answer": answer, "sources": [(d.metadata.get("page", 0) + 1, round(float(s), 3), d.page_content) for d, s in hits]}

def show(result):
    print("ANSWER:\n" + result["answer"] + "\n\nRETRIEVED CONTEXT:")
    for page, score, text in result["sources"]:
        print(f"  [p.{page}] (distance {score})  {text[:220].replace(chr(10), ' ')}{'...' if len(text) > 220 else ''}")


def main():
    pages = load_pages(); chunks = make_chunks(pages, 400, 60); emb = get_embeddings()
    vs = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True) if os.path.isdir(INDEX_DIR) else build_vectorstore(chunks, emb)
    llm = get_llm()
    if len(sys.argv) > 1:
        show(ask(" ".join(sys.argv[1:]), vs, llm)); return
    print("Ask about the M&A Playbook (blank line to quit).")
    while (q := input("\n> ").strip()):
        show(ask(q, vs, llm))

if __name__ == "__main__":
    main()
