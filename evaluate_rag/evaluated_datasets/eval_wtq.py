"""eval_wtq.py -- Comprehensive RAG Evaluation on WikiTableQuestions (WTQ).

Dataset  : stanfordnlp/wikitablequestions
Task     : Free-form QA over Wikipedia tables

Usage (env vars)
----------------
WTQ_SPLIT      = split to evaluate        (default: pristine-unseen-tables)
EVAL_SIZE      = max queries, 0=all        (default: 0)
WTQ_GENERATOR  = generator model           (default: gemini-3.1-flash-lite)
WTQ_JUDGE      = judge model               (default: gemini-3.1-flash-lite)

Output
------
evaluate_rag/eval_report_wtq.html   per-query dual-k HTML report
evaluate_rag/eval_report_wtq.json   raw metrics JSON

Metrics
-------
Retrieval (local, no API):    NDCG@10 | Recall@5 | Context Precision
Generation (RAGAS, LLM):      Faithfulness | Answer Relevancy
Answer Correctness (local):   Exact Match | F1 | Contains-Gold
"""

from __future__ import annotations

import asyncio, json, math, os, re, sys, time
from datetime import datetime

import dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
import html as html_mod


# ── paths & config ────────────────────────────────────────────────────────────
_DIR    = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_ROOT   = os.path.abspath(os.path.join(_DIR, ".."))
dotenv.load_dotenv(dotenv_path=os.path.join(_ROOT, ".env"))

GKEY     = os.getenv("GOOGLE_API_KEY", "").strip()
if not GKEY:
    sys.exit("[ERROR] GOOGLE_API_KEY not found in .env")

WTQ_DS   = "stanfordnlp/wikitablequestions"
WTQ_SPLIT = os.getenv("WTQ_SPLIT", "pristine-unseen-tables")
EVAL_SIZE = int(os.getenv("EVAL_SIZE", "0"))
OUT_HTML  = os.path.join(_DIR, "eval_report_wtq.html")
OUT_JSON  = os.path.join(_DIR, "eval_report_wtq.json")
CHROMA_DB = os.path.join(_ROOT, "chroma_db_wtq")

CSIZE, COVER  = 800, 100
EMODEL = "bge-m3"
RK, B3W, V3W, RTHR = 12, 0.3, 0.7, 0.85
GEN_M  = os.getenv("WTQ_GENERATOR", "gemini-3.1-flash-lite")
JUD_M  = os.getenv("WTQ_JUDGE",     "gemini-3.1-flash-lite")
K_VALS = [3, 5]
PAUSE  = 4

gen_llm = ChatGoogleGenerativeAI(model=GEN_M, google_api_key=GKEY, temperature=0.2)
jud_llm = ChatGoogleGenerativeAI(model=JUD_M, google_api_key=GKEY, temperature=0.0)
emb     = OllamaEmbeddings(model=EMODEL)

# Parsed WTQ examples cache (filled by index_wtq, consumed by main eval loop)
_EXAMPLES_CACHE = []


# ══════════════════════════════════════════════════════════════════════════════
#  TABLE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _dec(cell):
    # WTQ CSV/TSV escape sequences
    cell = cell.replace('\\"', '"')
    cell = cell.replace("\\n", "\n")
    cell = cell.replace("\\\\", "\\")
    cell = cell.replace("\\p", "|")
    return cell


def _ws(text):
    return set(
        w.strip(".,;:()[]-*").lower()
        for w in text.split()
        if len(w.strip(".,;:()[]-*")) > 1
    )


def table_txt(t, mx_rows=300):
    hdr  = [_dec(str(c)) for c in t.get("header", [])]
    rows = [[_dec(str(c)) for c in r] for r in t.get("rows", [])[:mx_rows]]
    name = _dec(str(t.get("name", "?")))
    nc   = len(hdr) if hdr else (len(rows[0]) if rows else 0)
    widths = [len(hdr[c]) for c in range(nc)]
    for r in rows:
        for c in range(min(nc, len(r))):
            widths[c] = max(widths[c], len(r[c]))
    parts = ["TABLE: " + name, "COLUMNS: " + " | ".join(hdr), "ROWS (" + str(len(rows)) + "):"]
    for i, r in enumerate(rows):
        pad = "  |  ".join(r[c].ljust(widths[c]) for c in range(min(nc, len(r))))
        parts.append("R" + str(i + 1) + ": " + pad)
    return "\n".join(parts)


def table_flat(t, mx_rows=300):
    hdr  = [_dec(str(c)) for c in t.get("header", [])]
    rows = t.get("rows", [])[:mx_rows]
    name = _dec(str(t.get("name", "?")))
    parts = ["table: " + name, "columns: " + " ".join(hdr)]
    for i, r in enumerate(rows):
        parts.append("row" + str(i + 1) + ": " + " | ".join(_dec(str(c)) for c in r))
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

def f_red(docs, thr=RTHR):
    uniq = []
    for d in docs:
        w = _ws(d.page_content)
        red = False
        for u in uniq:
            uw = _ws(u.page_content)
            if not w or not uw:
                continue
            if len(w & uw) / min(len(w), len(uw)) > thr:
                red = True
                break
        if not red:
            uniq.append(d)
    return uniq


class RerankRet:
    def __init__(self, base, ce, top=RK):
        self.base = base
        self.ce   = ce
        self.top  = top

    def invoke(self, q):
        docs = self.base.invoke(q)
        if not docs:
            return []
        seen = set()
        uniq = []
        for d in docs:
            if d.page_content not in seen:
                seen.add(d.page_content)
                uniq.append(d)
        if len(uniq) <= 1:
            return uniq[: self.top]
        scores = self.ce.predict([[q, d.page_content] for d in uniq])
        return [d for d, _ in sorted(zip(uniq, scores), key=lambda x: x[1], reverse=True)[: self.top]]


def build_hybrid(chunks, ce):
    bm = BM25Retriever.from_documents(chunks)
    bm.k = RK
    vt = Chroma(
        persist_directory=CHROMA_DB,
        embedding_function=emb,
        collection_name="wtq_eval",
    ).as_retriever(search_type="similarity", search_kwargs={"k": RK})
    return RerankRet(
        EnsembleRetriever(retrievers=[bm, vt], weights=[B3W, V3W]), ce, top=RK
    )


def get_vs():
    return Chroma(
        persist_directory=CHROMA_DB, embedding_function=emb, collection_name="wtq_eval"
    )


def clr_vs(vs):
    try:
        n = vs._collection.count()
        if n:
            print("[INDEX] Clearing " + str(n) + " existing chunks...")
            while True:
                ids = vs._collection.get(limit=500).get("ids", [])
                if not ids:
                    break
                vs.delete(ids=ids)
    except Exception as exc:
        print("[WARN] clr_vs: " + str(exc))


# ══════════════════════════════════════════════════════════════════════════════
#  INDEXING
# ══════════════════════════════════════════════════════════════════════════════

# ── WTQ raw data (GitHub release) ─────────────────────────────────────────────
WTQ_ZIP_URL = "https://github.com/ppasupat/WikiTableQuestions/releases/download/v1.0.2/WikiTableQuestions-1.0.2-compact.zip"
WTQ_CACHE  = os.path.join(_ROOT, "wtq_raw")


def _download_wtq():
    """Download + extract WTQ raw release once into WTQ_CACHE."""
    import zipfile
    os.makedirs(WTQ_CACHE, exist_ok=True)
    data_dir = os.path.join(WTQ_CACHE, "WikiTableQuestions")
    if os.path.isdir(data_dir) and os.path.isdir(os.path.join(data_dir, "data")):
        print("[INDEX] WTQ raw data already present at " + data_dir)
        return data_dir
    zip_path = os.path.join(WTQ_CACHE, "wtq.zip")
    if not os.path.exists(zip_path):
        print("[INDEX] Downloading WTQ release from GitHub...")
        import urllib.request
        urllib.request.urlretrieve(WTQ_ZIP_URL, zip_path)
        print("[INDEX] Downloaded: " + zip_path)
    print("[INDEX] Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(WTQ_CACHE)
    print("[INDEX] Extracted to " + data_dir)
    return data_dir


def _read_tsv_table(table_rel_path, root_dir):
    """Read a WTQ table TSV file -> {header, rows, name}."""
    tsv_path = os.path.join(root_dir, table_rel_path)
    # WTQ stores tables as .csv; fall back to .tsv if present
    if not os.path.exists(tsv_path):
        alt = tsv_path[:-4] + ".tsv"
        if os.path.exists(alt):
            tsv_path = alt
        else:
            return None
    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            vals = [_dec(v) for v in line.rstrip("\n").split("\t")]
            rows.append(vals)
    if not rows:
        return None
    return {"header": rows[0], "rows": rows[1:], "name": table_rel_path}


# Map dataset split name -> wtq data file name
_SPLIT_FILES = {
    "pristine-unseen-tables": "pristine-unseen-tables.tsv",
    "pristine-seen-tables": "pristine-seen-tables.tsv",
    "train":                "training.tsv",
    "training":             "training.tsv",
    "random-split-1":      "random-split-1-test.tsv",
}


def index_wtq(split, limit=0):
    root = _download_wtq()
    data_dir = os.path.join(root, "data")

    fname = _SPLIT_FILES.get(split, split + ".tsv" if not split.endswith(".tsv") else split)
    data_file = os.path.join(data_dir, fname)
    if not os.path.exists(data_file):
        # try finding any matching tsv
        cands = [f for f in os.listdir(data_dir) if split.replace("-", "") in f.replace("-", "")]
        if cands:
            data_file = os.path.join(data_dir, sorted(cands)[0])
        else:
            raise FileNotFoundError("WTQ data file not found for split '" + split + "' in " + data_dir)

    print("[INDEX] Reading questions from " + os.path.basename(data_file))
    examples = []  # list of dict: id, question, answers, table_path
    with open(data_file, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            rec = {header[i]: parts[i] for i in range(min(len(parts), len(header)))}
            qid   = rec.get("id", "")
            utt   = rec.get("utterance", "")
            ctx   = rec.get("context", "")
            tval  = rec.get("targetValue", "")
            if not utt or not ctx:
                continue
            answers = [a for a in tval.split("|")] if tval else []
            examples.append({"id": qid, "question": utt, "answers": answers, "table_path": ctx})

    N = len(examples)
    if limit and limit < N:
        examples = examples[:limit]
        N = limit
    print("[INDEX] Loaded " + str(N) + " examples from split '" + split + "'")

    # Build unique table path list
    rmap = {}
    for ex in examples:
        tp = ex["table_path"]
        # read table to get row count
        tbl = _read_tsv_table(tp, root)
        if tbl:
            rmap[tp] = len(tbl["rows"])
        else:
            rmap[tp] = 0

    tnames = sorted(rmap.keys())
    print("[INDEX] " + str(len(tnames)) + " unique tables found")

    raw = []
    for tp in tnames:
        tbl = _read_tsv_table(tp, root)
        if not tbl:
            continue
        raw.append(
            Document(
                page_content=table_txt(tbl) + "\n\n" + table_flat(tbl),
                metadata={
                    "source": tp,
                    "table_name": _dec(tp),
                    "n_rows": rmap[tp],
                },
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CSIZE, chunk_overlap=COVER, separators=["\nR", "\n", " | ", " "]
    )
    chunks = splitter.split_documents(raw)
    print("[INDEX] " + str(len(raw)) + " tables -> " + str(len(chunks)) + " chunks")

    vs = get_vs()
    clr_vs(vs)
    t0 = time.perf_counter()
    vs.add_documents(chunks)
    print("[INDEX] Indexed in " + str(round(time.perf_counter() - t0, 1)) + "s")

    # Store examples for the eval loop to consume without re-reading
    _EXAMPLES_CACHE.clear()
    _EXAMPLES_CACHE.extend(examples)
    return chunks, tnames, len(chunks)


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS — Retrieval
# ══════════════════════════════════════════════════════════════════════════════

def ndcg_at(docs, targets, k=10):
    rel = [1 if d.metadata.get("source") in targets else 0 for d in docs[:k]]
    if not rel:
        return 0.0
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(sum(rel), k)))
    return dcg / idcg if idcg else 0.0


def recall_at(docs, targets):
    if not targets:
        return 0.0
    hit = sum(1 for t in targets if any(d.metadata.get("source") == t for d in docs))
    return hit / len(targets)


def ctx_prec(docs, targets):
    if not docs:
        return 0.0
    rel, ps = 0, 0.0
    for i, d in enumerate(docs, 1):
        if d.metadata.get("source") in targets:
            rel += 1
            ps  += rel / i
    return ps / rel if rel else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS — Answer Correctness (local, no API)
# ══════════════════════════════════════════════════════════════════════════════

def _n(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()


def exact_match(pred, gold):
    return _n(pred) == _n(gold)


def tok_f1(pred, gold):
    pt = set(_n(pred).split())
    gt = set(_n(gold).split())
    if not pt and not gt:
        return 1.0
    inter = pt & gt
    pr = len(inter) / len(pt) if pt else 0.0
    rc = len(inter) / len(gt) if gt else 0.0
    return 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0


def contains_ans(pred, gold):
    return _n(gold) in _n(pred)


# ══════════════════════════════════════════════════════════════════════════════
#  GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _extract(content):
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "thinking" or "thinking" in p:
                    continue
                parts.append(p.get("text", str(p)))
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(content)


def _is_refusal(text):
    t = text.lower()
    return any(
        p in t
        for p in [
            "cannot answer",
            "does not contain",
            "no information",
            "not mentioned",
            "not discussed",
            "not provide information",
            "i do not know",
            "i am sorry",
            "insufficient context",
            "cannot be answered",
            "is not mentioned in",
        ]
    )


async def run_gen(query, ctx):
    sys_p = (
        "You are a precise table-question answering assistant.\n"
        "Answer using ONLY the information in the provided RAG context (Wikipedia table excerpts).\n"
        "If the context does not contain the answer, say exactly: "
        "'I cannot answer from the provided table data.'\n"
        "Give a concise answer. Do not explain reasoning. State only the final answer value."
    )
    try:
        resp = await gen_llm.ainvoke(
            [
                HumanMessage(content=sys_p),
                HumanMessage(
                    content="Retrieved table data:\n" + ctx + "\n\nQuestion: " + query + "\n\nAnswer:"
                ),
            ]
        )
        return _extract(resp.content).strip()
    except Exception as exc:
        return "[GENERATION ERROR] " + str(exc)


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION  (LLM judge -- RAGAS-style)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_j(raw):
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    for s in [
        m.group(0),
        m.group(0).replace("'", '"'),
        re.sub(r",\s*([\]}])", r"\1", m.group(0).replace("'", '"')),
    ]:
        try:
            return json.loads(s)
        except Exception:
            continue
    return None


async def eval_gen(query, ctx, gen, ref):
    ctx_s = ctx[:4000] if ctx else "(empty -- no data retrieved)"
    if _is_refusal(gen):
        return {"faithfulness": 1.0, "answer_relevancy": 1.0, "reasoning": "Correctly abstained."}
    prompt = (
        "You are an objective RAG evaluation judge. Score each metric 0.0 to 1.0.\n\n"
        "QUESTION: " + query + "\n\n"
        "RETRIEVED TABLE CONTEXT (first 4000 chars):\n" + ctx_s + "\n\n"
        "GENERATED ANSWER: " + gen + "\n\n"
        "REFERENCE ANSWER: " + ref + "\n\n"
        "FAITHFULNESS: Are ALL claims in the generated answer directly supported by the RETRIEVED CONTEXT?\n"
        "  1.0 = every claim grounded | 0.5 = partially | 0.0 = unsupported or fabricated\n\n"
        "ANSWER_RELEVANCY: Does the generated answer directly address the original QUESTION?\n"
        "  1.0 = fully addresses | 0.5 = partially | 0.0 = off-topic or evasive\n\n"
        "Respond ONLY with this JSON (no markdown):\n"
        '{"faithfulness": 0.0, "answer_relevancy": 0.0, "reasoning": "one sentence"}'
    )
    try:
        resp = await jud_llm.ainvoke([HumanMessage(content=prompt)])
        parsed = _parse_j(_extract(resp.content).strip())
        if parsed:
            return {
                "faithfulness": max(0.0, min(1.0, float(parsed.get("faithfulness", 0.5)))),
                "answer_relevancy": max(
                    0.0, min(1.0, float(parsed.get("answer_relevancy", 0.5)))
                ),
                "reasoning": str(parsed.get("reasoning", "")),
            }
    except Exception as exc:
        print("[JUDGE ERROR] " + str(exc))
    return {"faithfulness": 0.5, "answer_relevancy": 0.5, "reasoning": "Judge failed."}


# ══════════════════════════════════════════════════════════════════════════════
#  HTML HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _badge(ok, yes="PASS", no="FAIL"):
    c = "#22c55e" if ok else "#ef4444"
    return (
        '<span style="background:'
        + c
        + ";color:#fff;padding:2px 10px;border-radius:12px;"
        + "font-size:0.78em;font-weight:600\">"
        + (yes if ok else no)
        + "</span>"
    )


def _sb(sc):
    c = "#22c55e" if sc >= 0.7 else ("#f59e0b" if sc >= 0.4 else "#ef4444")
    return (
        '<span style="background:'
        + c
        + ";color:#fff;padding:2px 8px;border-radius:12px;"
        + "font-size:0.78em;font-weight:600\">"
        + f"{sc:.2f}"
        + "</span>"
    )


def _cell(txt, tag="td", extra=""):
    return "<" + tag + " " + extra + ">" + txt + "</" + tag + ">"


def _th(txt, **kw):
    sty = kw.get("sty", "")
    return _cell(txt, tag="th", extra='style="' + sty + '"')


def _td(txt, sty="", colspan=0):
    extra = 'style="' + sty + '"'
    c = ' colspan="' + str(colspan) + '"' if colspan else ""
    return "<td " + extra + c + ">" + txt + "</td>"


def save_html(results, S, path, split, NQ, NT, NC):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def card(title, color):
        tot = max(S["total"], 1)
        rows = []
        rows.append(
            '<tr><td colspan="2" style="padding:6px 0;font-weight:700;color:'
            + color
            + ';font-size:0.85em;text-transform:uppercase">Retrieval (BEIR)</td></tr>'
        )
        rows.append(
            '<tr><td style="padding:4px 0 4px 12px;color:#4b5563">NDCG@10</td>'
            '<td style="text-align:right;font-weight:700">'
            + f"{S['ndcg_10']/tot:.3f}</td></tr>"
        )
        rows.append(
            '<tr><td style="padding:4px 0 4px 12px;color:#4b5563">Recall@5</td>'
            '<td style="text-align:right;font-weight:700">'
            + f"{S['recall_5']/tot*100:.1f}%</td></tr>"
        )
        rows.append(
            '<tr><td style="padding:4px 0 4px 12px;color:#4b5563">Context Precision</td>'
            '<td style="text-align:right;font-weight:700">'
            + f"{S['ctx_prec']/(2*tot):.3f}</td></tr>"
        )
        rows.append(
            '<tr><td colspan="2" style="padding:6px 0;font-weight:700;color:'
            + color
            + ';font-size:0.85em;text-transform:uppercase">Generation (RAGAS)</td></tr>'
        )
        rows.append(
            '<tr><td style="padding:4px 0 4px 12px;color:#4b5563">Faithfulness</td>'
            '<td style="text-align:right;font-weight:700;color:#22c55e">'
            + f"{S['faith']/(2*tot):.3f}</td></tr>"
        )
        rows.append(
            '<tr><td style="padding:4px 0 4px 12px;color:#4b5563">Answer Relevancy</td>'
            '<td style="text-align:right;font-weight:700;color:#6366f1">'
            + f"{S['ans_rel']/(2*tot):.3f}</td></tr>"
        )
        rows.append(
            '<tr><td colspan="2" style="padding:6px 0;font-weight:700;color:'
            + color
            + ';font-size:0.85em;text-transform:uppercase">Correctness</td></tr>'
        )
        rows.append(
            '<tr><td style="padding:4px 0 4px 12px;color:#4b5563">Exact Match</td>'
            '<td style="text-align:right;font-weight:700">'
            + f"{S['em']/(2*tot)*100:.1f}%</td></tr>"
        )
        rows.append(
            '<tr><td style="padding:4px 0 4px 12px;color:#4b5563">Avg F1</td>'
            '<td style="text-align:right;font-weight:700">'
            + f"{S['f1']/(2*tot):.3f}</td></tr>"
        )
        rows.append(
            '<tr><td style="padding:4px 0 4px 12px;color:#4b5563">Contains Gold</td>'
            '<td style="text-align:right;font-weight:700">'
            + f"{S['contains']/(2*tot)*100:.1f}%</td></tr>"
        )
        for k in [3, 5]:
            ks = str(k)
            ret_s = f"{S['t_ret_'+ks]/tot:.2f}s"
            gen_s = f"{S['t_gen_'+ks]/tot:.2f}s"
            evl_s = f"{S['t_eval_'+ks]/tot:.2f}s"
            tot_s = f"{S['t_tot_'+ks]/tot:.2f}s"
            rows.append(
                '<tr><td colspan="2" style="padding:4px 0;font-weight:600;color:#1e293b;font-size:0.88em;'
                + 'border-top:1px dashed #e2e8f0\">k='
                + ks
                + " Latency (Ret / Gen / Eval / Total)</td></tr>"
            )
            rows.append(
                '<tr><td style="padding:2px 0 2px 16px;color:#6b7280;font-size:0.85em\">k='
                + ks
                + "</td>"
                '<td style="text-align:right;font-size:0.85em;color:#4b5563\">'
                + ret_s
                + " / "
                + gen_s
                + " / "
                + evl_s
                + " / <strong>"
                + tot_s
                + "</strong></td></tr>"
            )
        return (
            '<div style="background:#fff;border-radius:12px;padding:20px 24px;'
            + 'box-shadow:0 1px 8px #0001;border-top:4px solid '
            + color
            + ";flex:1;min-width:340px\">"
            '<h3 style="margin:0 0 4px;color:'
            + color
            + '">'
            + title
            + "</h3>"
            '<p style="color:#6b7280;font-size:0.85em;margin-bottom:16px\">'
            "WTQ "
            + split
            + " | "
            + str(NT)
            + " tables | "
            + str(NC)
            + " chunks</p>"
            '<table style="width:100%;border-collapse:collapse;font-size:0.9em\">'
            '<tr style="border-bottom:1px solid #f3f4f6\">'
            '<th style="text-align:left;padding:6px 0\">Metric</th>'
            '<th style="text-align:right\">Score</th></tr>'
            + "".join(rows)
            + "</table></div>"
        )

    card_html = card("All " + str(NQ) + " Queries", "#4f46e5")

    q_rows = ""
    for r in results:
        qe   = html_mod.escape(r["question"])
        gs   = html_mod.escape(", ".join(r["gold_answers"]))
        tns  = html_mod.escape(r["source_table"])
        k3   = r["k3"]
        k5   = r["k5"]

        # --- metric rows ---
        mr  = '<tr style="border-top:1px solid #e5e7eb;background:#fafafa">'
        mr += _td("Metric (k=3 / k=5)", sty="padding:6px 12px;color:#374151;font-weight:700;font-size:0.82em", colspan=3)
        mr += "</tr>"
        mr += '<tr style="border-top:1px solid #e5e7eb">'
        mr += _td("NDCG@10", "padding:8px 12px;color:#6b7280;font-size:0.85em")
        mr += _td(f"{k3['ndcg_10']:.3f}", "text-align:center;font-weight:600")
        mr += _td(f"{k5['ndcg_10']:.3f}", "text-align:center;font-weight:600")
        mr += "</tr>"
        mr += '<tr style="border-top:1px solid #e5e7eb">'
        mr += _td("Recall@5 HIT", "padding:8px 12px;color:#6b7280;font-size:0.85em")
        mr += _td(_badge(k3["recall_5"] > 0), "text-align:center")
        mr += _td(_badge(k5["recall_5"] > 0), sty="text-align:center")
        mr += "</tr>"
        mr += '<tr style="border-top:1px solid #e5e7eb;background:#fafafa">'
        mr += _td("Generation", "padding:6px 12px;color:#374151;font-weight:700;font-size:0.82em", colspan=3)
        mr += "</tr>"
        mr += '<tr style="border-top:1px solid #e5e7eb">'
        mr += _td("Faithfulness", "padding:8px 12px;color:#6b7280;font-size:0.85em")
        mr += _td(_sb(k3["faithfulness"]), sty="text-align:center")
        mr += _td(_sb(k5["faithfulness"]), sty="text-align:center")
        mr += "</tr>"
        mr += '<tr style="border-top:1px solid #e5e7eb">'
        mr += _td("Ans Relevancy", "padding:8px 12px;color:#6b7280;font-size:0.85em")
        mr += _td(_sb(k3["answer_relevancy"]), sty="text-align:center")
        mr += _td(_sb(k5["answer_relevancy"]), sty="text-align:center")
        mr += "</tr>"
        mr += '<tr style="border-top:1px solid #e5e7eb;background:#fafafa">'
        mr += _td("Correctness", "padding:6px 12px;color:#374151;font-weight:700;font-size:0.82em", colspan=3)
        mr += "</tr>"
        mr += '<tr style="border-top:1px solid #e5e7eb">'
        mr += _td("Exact Match", "padding:8px 12px;color:#6b7280;font-size:0.85em")
        mr += _td(_badge(k3["exact_match"]), sty="text-align:center;font-weight:600")
        mr += _td(_badge(k5["exact_match"]), sty="text-align:center;font-weight:600")
        mr += "</tr>"
        mr += '<tr style="border-top:1px solid #e5e7eb">'
        mr += _td("F1 (k3 / k5)", sty="padding:8px 12px;color:#6b7280;font-size:0.80em")
        mr += _td(f"{k3['f1']:.3f}", "text-align:center;font-weight:600")
        mr += _td(f"{k5['f1']:.3f}", "text-align:center;font-weight:600")
        mr += "</tr>"
        mr += '<tr style="border-top:1px solid #e5e7eb">'
        mr += _td("Contains Gold (k3/k5)", sty="padding:8px 12px;color:#6b7280;font-size:0.80em")
        mr += _td(_badge(k3["contains_gold"]), sty="text-align:center")
        mr += _td(_badge(k5["contains_gold"]), sty="text-align:center")
        mr += "</tr>"
        mr += '<tr style="border-top:1px solid #e5e7eb;background:#fafafa">'
        mr += _td("Latency", "padding:6px 12px;color:#374151;font-weight:700;font-size:0.82em", colspan=3)
        mr += "</tr>"
        for k in [3, 5]:
            ks = str(k)
            mr += '<tr style="border-top:1px solid #e5e7eb">'
            mr += _td("k=" + ks, "padding:8px 12px;color:#6b7280;font-size:0.85em")
            mr += _td(
                f"{r['ret_'+ks]:.2f}s / {r['gen_'+ks]:.2f}s / {r['eval_'+ks]:.2f}s / "
                + "<strong>" + f"{r['tot_'+ks]:.2f}s</strong>", "text-align:center;font-size:0.85em;color:#4b5563",
                colspan=2,
            )
            mr += "</tr>"

    rsn = html_mod.escape(k3.get("reasoning", ""))
    gn  = html_mod.escape(r["generated_k3"][:350])
    rn  = html_mod.escape(r["reference"][:350])
    jh  = (
        '<div style="background:#fef3c7;border:1px solid #fde68a;border-radius:8px;'
        + 'padding:8px 12px;font-size:0.82em;color:#92400e;margin-bottom:12px\">'
        + "<strong>Judge:</strong> "
        + rsn
        + "</div>"
        if rsn
        else ""
    )

    q_rows += (
        '<details style="margin-bottom:12px;border:1px solid #e5e7eb;border-radius:10px;overflow:hidden\"> '
        '<summary style="padding:14px 18px;cursor:pointer;background:#f9fafb;display:flex;'
        'align-items:center;gap:10px;list-style:none\">'
        '<span style="font-weight:600;color:#111;flex:1\">Q'
        + str(r["idx"])
        + ". "
        + qe
        + "</span>"
        '<span style="font-size:0.75em;color:#6b7280\">Table: '
        + tns
        + "</span>"
        + _sb(k3["faithfulness"])
        + "</summary>"
        '<div style="padding:16px 20px;background:#fff\">'
        '<p style="font-size:0.82em;color:#6b7280;margin-bottom:8px\">'
        "<strong>Gold:</strong> "
        + gs
        + ' &nbsp;|&nbsp; <strong>Table:</strong> '
        + tns
        + "</p>"
        '<table style="width:100%;border-collapse:collapse;margin-bottom:16px\"><thead><tr style="background:#f3f4f6\">'
        '<th style="padding:8px 12px;text-align:left;color:#374151;font-size:0.85em\">Metric</th>'
        '<th style="padding:8px;text-align:center;color:#6366f1;font-size:0.85em\">k=3</th>'
        '<th style="padding:8px;text-align:center;color:#8b5cf6;font-size:0.85em\">k=5</th>'
        "</tr></thead><tbody>"
        + mr
        + "</tbody></table>"
        + jh
        + '<div style="display:flex;gap:16px;flex-wrap:wrap\">'
        '<div style="flex:1;min-width:260px\">'
        '<div style="font-size:0.78em;font-weight:700;color:#6b7280;text-transform:uppercase;margin-bottom:4px\">'
        "Reference Answer</div>"
        '<div style="background:#f0fdf4;border:1px solid #bbf7d0;color:#166534;'
        'border-radius:8px;padding:10px;font-size:0.85em;white-space:pre-wrap\">'
        + rn
        + "</div></div>"
        '<div style="flex:1;min-width:260px\">'
        '<div style="font-size:0.78em;font-weight:700;color:#6366f1;text-transform:uppercase;margin-bottom:4px\">'
        "Generated (k=3)</div>"
        '<div style="background:#eff6ff;border:1px solid #bfdbfe;color:#1e40af;'
        'border-radius:8px;padding:10px;font-size:0.85em;white-space:pre-wrap\">'
        + gn
        + "</div></div></div></div></details>"
    )

    css = (
        '* {box-sizing:border-box;margin:0;padding:0}'
        'body {font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;'
        'background:#f8fafc;color:#1a1a2e;padding:32px 24px}'
        "h1 {font-size:1.8em;font-weight:800;margin-bottom:4px}"
        "h2 {font-size:1.1em;font-weight:700;margin:28px 0 12px;color:#1e293b;"
        "border-bottom:2px solid #e2e8f0;padding-bottom:6px}"
        ".subtitle {color:#6b7280;margin-bottom:28px;font-size:0.92em}"
        ".cards {display:flex;gap:16px;flex-wrap:wrap;margin-bottom:32px}"
        "details>summary::-webkit-details-marker {display:none}"
        "details>summary::before {content:'\\25b6';margin-right:8px;font-size:0.75em;"
        "color:#9ca3af;transition:transform .2s}"
        "details[open]>summary::before {transform:rotate(90deg)}"
    )

    doc = (
        "<!DOCTYPE html><html lang='en'><head>"
        '<meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>WTQ RAG Evaluation Report</title>"
        "<style>" + css + "</style>"
        "</head><body>"
        "<h1>WTQ RAG Evaluation Report</h1>"
        '<p class="subtitle">'
        "Dataset: <strong>stanfordnlp/wikitablequestions</strong>"
        " &middot; Split: <strong>"
        + html_mod.escape(WTQ_SPLIT)
        + "</strong>"
        " &middot; Queries: <strong>"
        + str(NQ)
        + "</strong>"
        " &middot; Tables: <strong>"
        + str(NT)
        + "</strong>"
        " &middot; Chunks: <strong>"
        + str(NC)
        + "</strong><br>"
        "Generator: <strong>" + GEN_M + "</strong>"
        " &middot; Judge: <strong>" + JUD_M + "</strong>"
        " &middot; Generated: <strong>" + ts + "</strong>"
        "</p>"
        '<h2>Summary Metrics</h2>'
        '<div class="cards">' + card_html + "</div>"
        "<h2>Per-Query Results (" + str(NQ) + " queries, k=3 vs k=5)</h2>"
        + q_rows
        + "</body></html>"
    )
    with open(path, "w", encoding="utf-8") as fout:
        fout.write(doc)
    print("[REPORT] HTML saved: " + path)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS for HTML rows
# ══════════════════════════════════════════════════════════════════════════════

def _tr(body="", kv="", colspan=0):
    c = ' colspan="' + str(colspan) + '"' if colspan else ""
    return '<tr style="' + kv + '"' + c + ">" + body + "</tr>"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def newS():
    return {
        "ndcg_10": 0.0,
        "recall_5": 0.0,
        "ctx_prec": 0.0,
        "faith": 0.0,
        "ans_rel": 0.0,
        "em": 0,
        "f1": 0.0,
        "contains": 0,
        "total": 0,
        "t_ret_3": 0.0,
        "t_ret_5": 0.0,
        "t_gen_3": 0.0,
        "t_gen_5": 0.0,
        "t_eval_3": 0.0,
        "t_eval_5": 0.0,
        "t_tot_3": 0.0,
        "t_tot_5": 0.0,
    }


async def main():
    split = WTQ_SPLIT
    limit = EVAL_SIZE

    print("")
    print("=" * 80)
    print("  WTQ COMPREHENSIVE RAG EVALUATION")
    print("  Dataset  : " + WTQ_DS)
    print("  Split    : " + split)
    print("  Eval size: " + ("all" if not limit else str(limit)))
    print("  Generator: " + GEN_M + "  |  Judge: " + JUD_M)
    print("=" * 80)

    # 1. load dataset (raw files)
    print("\n[1/6] Loading WTQ dataset (raw GitHub release)...")
    _download_wtq()  # ensure raw data present
    N  = len(_EXAMPLES_CACHE)
    print("       " + str(N) + " examples available for split '" + split + "'")

    # 2. index
    print("\n[2/6] Serialising and indexing tables into Chroma...")
    ce = CrossEncoder("BAAI/bge-reranker-v2-m3")
    print("[INFO] BGE-Reranker-v2-m3 loaded")
    chunks, tnames, nchunks = index_wtq(split, limit=limit)

    # 3. build retriever
    print("\n[3/6] Building hybrid BM25+Vector+BGE-Reranker retriever...")
    retriever = build_hybrid(chunks, ce)
    print("[OK] Retriever ready")

    # 4. evaluation loop
    print(f"\n[4/6] Evaluating {N} queries...")
    print("=" * 80)
    S   = newS()
    res = []

    for idx, ex in enumerate(_EXAMPLES_CACHE, 1):
        qtext = ex["question"]
        gold  = [str(a) for a in (ex["answers"] or [])]
        src   = _dec(str(ex["table_path"]))
        ref   = gold[0] if gold else ""

        print("\n[Q" + str(idx) + "/" + str(N) + "] " + repr(qtext))
        print("          Table=" + src + "  Gold=" + str(gold))

        pk    = {}
        lc    = {}
        gen_ans_cache = {}
        ndcgv = 0.0

        for k in K_VALS:
            t0 = time.perf_counter()
            raw = retriever.invoke(qtext)
            seen = set()
            uniq = []
            for d in raw:
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    uniq.append(d)
            filt  = f_red(uniq)
            final = filt[:k]
            ctx   = "\n\n---\n\n".join(d.page_content for d in final) if final else "(no data retrieved)"

            ndcgv = ndcg_at(uniq, [src], k=10)
            rc    = recall_at(final, [src])
            cp    = ctx_prec(final, [src])

            t1 = time.perf_counter()
            gen = await run_gen(qtext, ctx)
            dtg = time.perf_counter() - t1
            gen_ans_cache[k] = gen

            t2 = time.perf_counter()
            ragas = await eval_gen(qtext, ctx, gen, ref)
            dte   = time.perf_counter() - t2
            dtt   = time.perf_counter() - t0

            ems  = max(exact_match(gen, g) for g in gold) if gold else False
            f1v  = max(tok_f1(gen, g) for g in gold) if gold else 0.0
            cg   = any(contains_ans(gen, g) for g in gold) if gold else False

            pk[k] = {
                "ndcg_10": ndcgv,
                "recall_5": rc,
                "ctx_prec": cp,
                "faithfulness": ragas["faithfulness"],
                "answer_relevancy": ragas["answer_relevancy"],
                "exact_match": ems,
                "f1": f1v,
                "contains_gold": cg,
                "reasoning": ragas.get("reasoning", ""),
            }
            lc[k] = {"ret": t1 - t0, "gen": dtg, "eval": dte, "tot": dtt}

            z = S
            if k == 3:
                z["ndcg_10"] += ndcgv
            if k == 5:
                z["recall_5"] += rc
            z["ctx_prec"]   += cp
            z["faith"]      += ragas["faithfulness"]
            z["ans_rel"]    += ragas["answer_relevancy"]
            z["em"]         += int(ems)
            z["f1"]         += f1v
            z["contains"]   += int(cg)
            ks = str(k)
            z["t_ret_" + ks] += t1 - t0
            z["t_gen_" + ks] += dtg
            z["t_eval_" + ks] += dte
            z["t_tot_" + ks] += dtt

            print(
                "  [k="
                + str(k)
                + "] NDCG="
                + f"{ndcgv:.3f}"
                + " Rec="
                + f"{rc:.1f}"
                + " CtxP="
                + f"{cp:.3f}"
                + " Faith="
                + f"{ragas['faithfulness']:.2f}"
                + " AnsRel="
                + f"{ragas['answer_relevancy']:.2f}"
                + " EM="
                + str(ems)
                + " F1="
                + f"{f1v:.3f}"
                + " CF="
                + str(cg)
                + " | R="
                + f"{t1-t0:.2f}s"
                + " G="
                + f"{dtg:.2f}s"
                + " E="
                + f"{dte:.2f}s"
                + " T="
                + f"{dtt:.2f}s"
            )

        S["total"] += 1
        res.append(
            {
                "idx": idx,
                "question": qtext,
                "source_table": src,
                "gold_answers": gold,
                "reference": ref,
                "generated_k3": gen_ans_cache.get(3, ""),
                "generated_k5": gen_ans_cache.get(5, ""),
                "k3": pk[3],
                "k5": pk[5],
                "ret_3": lc[3]["ret"],
                "ret_5": lc[5]["ret"],
                "gen_3": lc[3]["gen"],
                "gen_5": lc[5]["gen"],
                "eval_3": lc[3]["eval"],
                "eval_5": lc[5]["eval"],
                "tot_3":  lc[3]["tot"],
                "tot_5":  lc[5]["tot"],
            }
        )
        if idx < N:
            await asyncio.sleep(PAUSE)

    # 5. summary
    t = max(S["total"], 1)
    print("\n" + "=" * 80)
    print("FINAL WTQ SUMMARY")
    print("=" * 80)
    print("  Queries      : " + str(S["total"]))
    print("  Tables       : " + str(len(tnames)))
    print("  Chunks       : " + str(nchunks))
    print("  NDCG@10      : " + f"{S['ndcg_10']/t:.3f}")
    print("  Recall@5     : " + f"{S['recall_5']/t*100:.1f}%")
    print("  Ctx Precision: " + f"{S['ctx_prec']/(2*t):.3f}")
    print("  Faithfulness : " + f"{S['faith']/(2*t):.3f}")
    print("  Ans Relevancy: " + f"{S['ans_rel']/(2*t):.3f}")
    print("  Exact Match  : " + f"{S['em']/(2*t)*100:.1f}%")
    print("  Avg F1       : " + f"{S['f1']/(2*t):.3f}")
    print("  Contains Gold: " + f"{S['contains']/(2*t)*100:.1f}%")
    for k in [3, 5]:
        ks = str(k)
        print(
            "  k="
            + ks
            + " | Ret="
            + f"{S['t_ret_'+ks]/t:.2f}s"
            + " Gen="
            + f"{S['t_gen_'+ks]/t:.2f}s"
            + " Eval="
            + f"{S['t_eval_'+ks]/t:.2f}s"
            + " Tot="
            + f"{S['t_tot_'+ks]/t:.2f}s"
        )
    print("=" * 80)

    # 6. save
    print("\n[6/6] Saving reports...")
    save_html(res, S, OUT_HTML, split, N, len(tnames), nchunks)

    lat = {}
    for k in [3, 5]:
        ks = str(k)
        lat["avg_retrieval_k" + ks] = S["t_ret_" + ks] / t
        lat["avg_generation_k" + ks] = S["t_gen_" + ks] / t
        lat["avg_eval_k" + ks]       = S["t_eval_" + ks] / t
        lat["avg_total_k" + ks]      = S["t_tot_" + ks] / t

    metrics = {
        "dataset": WTQ_DS,
        "split": split,
        "total_queries": N,
        "n_tables": len(tnames),
        "n_chunks": nchunks,
        "models": {"generator": GEN_M, "judge": JUD_M},
        "metrics": {
            "ndcg_10":         S["ndcg_10"] / t,
            "recall_5":        S["recall_5"] / t,
            "ctx_precision":   S["ctx_prec"] / (2 * t),
            "faithfulness":    S["faith"]    / (2 * t),
            "ans_relevancy":   S["ans_rel"]  / (2 * t),
            "exact_match_pct": S["em"]       / (2 * t) * 100,
            "avg_f1":          S["f1"]       / (2 * t),
            "contains_gold_pct": S["contains"] / (2 * t) * 100,
        },
        "latency": lat,
        "per_query": res,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fout:
        json.dump(metrics, fout, indent=2, ensure_ascii=False)
    print("       JSON saved: " + OUT_JSON)
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
