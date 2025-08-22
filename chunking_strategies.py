

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Try to import SemanticChunker (some envs may not have it)
try:
    from langchain_experimental.text_splitter import SemanticChunker
    HAS_SEMANTIC = True
except Exception:
    HAS_SEMANTIC = False

# Local, no-API embeddings for SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings


SAMPLE_TEXT = (
    "Problem: Long-context, multilingual question answering often fails when key evidence is scattered across sections and languages. "
    "Data: We curate a 12-language corpus with parallel passages, noisy OCR scans, and code-mixed queries to stress retrieval in real settings. "
    "Method: A hybrid retriever blends BM25 for lexical anchors with a contrastively trained dense encoder for semantic recall, then re-ranks with a lightweight cross-encoder. "
    "Indexing: We store sentence- and paragraph-level vectors with parent–child links, enabling dynamic expansion to recover lost context without inflating prompts. "
    "Reader: A compact long-context encoder–decoder consumes diverse snippets and cites exact spans, prioritizing faithfulness over verbosity. "
    "Training: We mine hard negatives across languages, apply MMR to reduce redundancy, and distill from a larger cross-encoder to keep latency low. "
    "Results: On three internal benchmarks we observe consistent gains in nDCG@10 and answer EM under a fixed 3k-token budget, with especially large wins on noisy PDFs. "
    "Ablations: Removing cross-lingual negatives harms recall most, while disabling dynamic expansion increases hallucinations despite similar token spend. "
    "Efficiency: Batchable re-ranking and quantized embeddings cut median latency without degrading attribution quality. "
    "Robustness: The system maintains answer quality under synonym swaps, spelling noise, and partial table extractions, though tables with merged headers remain challenging. "
    "Limitations: Performance dips on low-resource languages with scarce domain terms, and OCR artifacts can still mislead the reader. "
    "Ethics: We filter PII at chunk time and redact before indexing; user queries are anonymized and retained only for aggregate evaluation. "
    "Deployment: A guardrail layer detects out-of-scope questions and returns citations-only summaries when confidence is low. "
    "Future work: We plan retrieval-time translation, layout-aware table repair, and active learning loops that target the hardest queries."
)

SAMPLE_TEXT = (
    "Machine Learning: Split data into train/validation/test, fit a baseline, and tune hyperparameters to maximize F1 without overfitting. "
    "Deep Learning: Use a GPU to train a small Transformer with mixed precision and early stopping, then export to ONNX for fast inference. "
    "Generative AI: Serve the model behind a prompt API with RAG for context, monitor hallucinations and toxicity, and roll back if drift spikes."
)

def show(title, docs, max_chars=120):
    print(f"\n=== {title} — {len(docs)} chunks ===")
    for i, d in enumerate(docs, 1):
        text = d.page_content.replace("\n", " ")
        print(f"[{i:02d}] {text[:max_chars]}{'...' if len(text)>max_chars else ''}")

def recursive_split(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,     # try 200–400 for general text
        chunk_overlap=30,   # small overlap helps if facts cross boundaries
        # default separators are fine; you can pass your own list if needed
    )
    return splitter.create_documents([text])

def semantic_split(text: str):
    if not HAS_SEMANTIC:
        print("SemanticChunker not available. Install/upgrade langchain-experimental.")
        return []

    # Local embeddings (no API key). Good default: all-MiniLM-L6-v2
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Keep args minimal for compatibility across versions
    sem = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",   # split where meaning shifts
        breakpoint_threshold_amount=99.0,         # higher = fewer splits
        add_start_index=True,
    )
    return sem.create_documents([text])

if __name__ == "__main__":
    # 1) Recursive (size+overlap)
    rec_docs = recursive_split(SAMPLE_TEXT)
    show("RecursiveCharacterTextSplitter", rec_docs)

    # 2) Semantic (meaning-based)
    sem_docs = semantic_split(SAMPLE_TEXT)
    if sem_docs:
        show("SemanticChunker (HF local embeddings)", sem_docs)