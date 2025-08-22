# llm_chunker_simple.py
# Ask an LLM to segment text into contiguous, semantically similar chunks.

import os, json, re, argparse, textwrap
from typing import List, Dict, Optional

from langchain_openai import ChatOpenAI  # langchain>=0.2
from langchain_core.messages import SystemMessage, HumanMessage

def simple_sentence_split(text: str) -> List[str]:
    # Lightweight splitter that keeps punctuation with the sentence
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

PROMPT_TEMPLATE = """You are a precise text segmenter.
Goal: split the following sentences into CONTIGUOUS chunks so that sentences *within each chunk* are semantically similar and about the same topic.
Rules:
- Keep original order; no overlaps; cover ALL sentences.
- Prefer 2–6 chunks by default. If target_n is provided, produce exactly that many chunks.
- Output STRICT JSON: {{"chunks":[{{"start":int,"end":int,"label":str}}...]}}
  - 'start' and 'end' are INCLUSIVE 0-based sentence indices.
  - 'label' is a short topic name (e.g., "Pricing", "Auth", "Errors").

target_n: {target_n}

Sentences (indexed):
{indexed_sentences}

Return ONLY the JSON, no extra text.
"""

def ask_llm_to_chunk(sentences: List[str], target_n: Optional[int]=None, model_name: str="gpt-4o-mini") -> List[Dict]:
    llm = ChatOpenAI(model=model_name, temperature=0)
    indexed = "\n".join([f"[{i}] {s}" for i, s in enumerate(sentences)])
    prompt = PROMPT_TEMPLATE.format(
        target_n=("exactly " + str(target_n)) if target_n else "none",
        indexed_sentences=indexed
    )
    msgs = [SystemMessage(content="Follow instructions exactly and return valid JSON."),
            HumanMessage(content=prompt)]
    resp = llm.invoke(msgs).content.strip()

    # Try to extract JSON (in case the model adds stray text)
    m = re.search(r"\{[\s\S]*\}$", resp)
    raw = m.group(0) if m else resp
    data = json.loads(raw)  # will raise if invalid
    chunks = data["chunks"]
    # Basic validation
    n = len(sentences)
    for c in chunks:
        assert 0 <= c["start"] <= c["end"] < n, "index out of bounds"
    # Ensure coverage & contiguity (optional strict check)
    covered = []
    for c in chunks: covered.extend(list(range(c["start"], c["end"]+1)))
    assert sorted(covered) == list(range(n)), "chunks must cover all sentences exactly once"
    return chunks

def materialize_chunks(sentences: List[str], chunks: List[Dict]) -> List[Dict]:
    out = []
    for c in chunks:
        text = " ".join(sentences[c["start"]:c["end"]+1])
        out.append({"label": c.get("label",""), "start": c["start"], "end": c["end"], "text": text})
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, default="", help="Inline text to split")
    ap.add_argument("--text_file", type=str, help="Path to .txt file")
    ap.add_argument("--n", type=int, default=None, help="Exact number of chunks (optional)")
    args = ap.parse_args()

    if not args.text and not args.text_file:
        # Simple default: 3 sentences on ML/DL/GenAI (nice semantic contrast)
        args.text = (
            "Machine Learning: Split data into train/validation/test and tune hyperparameters to maximize F1 without overfitting. "
            "Deep Learning: Train a small Transformer with mixed precision on a GPU, then export to ONNX for fast inference. "
            "Generative AI: Serve a prompt API with RAG, monitor hallucinations and toxicity, and roll back if drift spikes."
        )

    text = args.text or open(args.text_file, "r", encoding="utf-8").read()
    sentences = simple_sentence_split(text)
    chunks = ask_llm_to_chunk(sentences, target_n=args.n)
    out = materialize_chunks(sentences, chunks)

    print("\n== Chunks ==")
    for i, c in enumerate(out, 1):
        print(f"[{i}] {c['label']}  ({c['start']}..{c['end']}): {c['text']}")