"""
retrieval/table_serialization.py — Row-by-row natural language table serialization.

Converts each row of a tabular source into a self-contained, descriptive sentence
that repeats the table title and column headers. Each row is emitted as a discrete
Document (no text-splitter, overlap 0) so that fragments of different rows never
bleed into each other and confuse BGE-M3.

Sources handled:
  * Tables embedded inside PDFs      — via load_pdf_table_documents()
"""

import os
from langchain_core.documents import Document


# Called in: multi_agent/retrieval/ingestion.py (load_and_index_documents)
def load_pdf_prose_and_tables(docs_dir: str) -> tuple[list[Document], list[Document]]:
    """Extract prose (excluding table content) and tables from every *.pdf in `docs_dir`.

    Uses pdfplumber to detect tables, filter out characters inside tables to get clean prose,
    and serialize table rows into discrete Document objects.
    """
    try:
        import pdfplumber
    except ImportError:
        print("[WARN] pdfplumber not installed — skipping PDF extraction.")
        return [], []

    prose_docs: list[Document] = []
    table_docs: list[Document] = []

    for fname in sorted(os.listdir(docs_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(docs_dir, fname)
        pdf_stem = os.path.splitext(fname)[0]
        with pdfplumber.open(fpath) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                tables = page.find_tables()
                
                # 1) Extract Tables first and record their bounding boxes
                prev_y1 = None
                table_bboxes = []
                for t_idx, t_obj in enumerate(tables, start=1):
                    table_bboxes.append(t_obj.bbox)
                    table_data = t_obj.extract()
                    title_text = _find_table_title(page, t_obj.bbox[1], prev_y1)
                    if title_text:
                        table_name = title_text
                    else:
                        table_name = f"{pdf_stem} table {t_idx}"
                    
                    table_docs.extend(
                        _rows_to_documents(
                            table_data,
                            table_name,
                            fpath,
                            {"page": page_idx, "pdf_table": t_idx},
                        )
                    )
                    prev_y1 = t_obj.bbox[3]

                # 2) Extract Prose by filtering out characters inside table bounding boxes
                def keep_outside_tables(obj):
                    if obj.get("object_type") == "char":
                        x0, top, x1, bottom = obj["x0"], obj["top"], obj["x1"], obj["bottom"]
                        for tx0, ttop, tx1, tbottom in table_bboxes:
                            if x0 >= tx0 and x1 <= tx1 and top >= ttop and bottom <= tbottom:
                                return False
                    return True

                prose_text = None
                try:
                    filtered_page = page.filter(keep_outside_tables)
                    prose_text = filtered_page.extract_text()
                except Exception:
                    pass

                if not prose_text or not prose_text.strip():
                    try:
                        prose_text = page.extract_text()
                    except Exception:
                        pass

                if prose_text and prose_text.strip():
                    prose_docs.append(
                        Document(
                            page_content=prose_text.strip(),
                            metadata={"source": fpath, "page": page_idx - 1},
                        )
                    )

        # Fallback: if pdfplumber produced 0 prose_docs and 0 table_docs for a PDF, try pypdf / pypdfloader
        if not prose_docs and not table_docs:
            try:
                from pypdf import PdfReader
                reader = PdfReader(fpath)
                for idx, p in enumerate(reader.pages):
                    txt = p.extract_text()
                    if txt and txt.strip():
                        prose_docs.append(Document(page_content=txt.strip(), metadata={"source": fpath, "page": idx}))
            except Exception as e:
                print(f"[WARN] pypdf fallback failed for {fpath}: {e}")

    return prose_docs, table_docs


# Called in: multi_agent/retrieval/table_serialization.py (load_pdf_prose_and_tables)
def _find_table_title(page, table_y0: float, prev_table_y1: float | None) -> str:
    """Find a descriptive title/caption for the table from the text above it."""
    above_words = [
        w for w in page.extract_words()
        if w["bottom"] <= table_y0 and (prev_table_y1 is None or w["top"] >= prev_table_y1)
    ]
    if not above_words:
        return ""

    # Group words into lines based on similar vertical top coordinate
    lines_dict = {}
    for w in above_words:
        grp = next((g for g in lines_dict if abs(w["top"] - g) < 4), w["top"])
        lines_dict.setdefault(grp, []).append(w)

    # Sort lines from top to bottom, and sort words within each line from left to right
    sorted_lines = [
        " ".join(w["text"] for w in sorted(words, key=lambda x: x["x0"])).strip()
        for _, words in sorted(lines_dict.items())
    ]
    sorted_lines = [line for line in sorted_lines if line]

    table_headers = [line for line in sorted_lines if "table" in line.lower()]
    if table_headers:
        return table_headers[-1]

    return " ".join(sorted_lines[-2:]) if sorted_lines else ""


# Called in: multi_agent/retrieval/table_serialization.py (_rows_to_documents)
def _rows_to_documents(rows: list[list], table_name: str, source: str,
                       base_meta: dict) -> list[Document]:
    """Turn raw pdfplumber rows (header + data) into serialized Documents.

    Cell text is whitespace-normalized (pdfplumber keeps newline-wrapped cells).
    If a column header is empty, it is given a generic 'Column N' label. If the
    first (subject) cell is empty, the first non-empty cell is used as subject.
    """
    if not rows:
        return []

    raw_header = [_clean_cell(h) for h in rows[0]]
    header = [
        raw_header[j] if raw_header[j] else f"Column {j + 1}"
        for j in range(len(raw_header))
    ]

    docs: list[Document] = []
    current_section = ""
    for i, row in enumerate(rows[1:], start=0):
        clean_cells = [_clean_cell(c) for c in row]
        if not any(clean_cells):
            continue  # skip fully empty rows

        # Track hierarchical category section headers (e.g., "2010", "2009", "Non-current assets")
        # A section header has text in the first column and no data in remaining columns
        first_cell = clean_cells[0]
        remaining_cells = clean_cells[1:]
        if len(clean_cells) > 1 and first_cell and not any(remaining_cells):
            current_section = first_cell
            continue

        # Map every column to its (cleaned) value; fall back to "" if missing.
        clean_row = {}
        for j in range(len(header)):
            if j < len(clean_cells):
                clean_row[header[j]] = clean_cells[j]
            else:
                clean_row[header[j]] = ""

        if not any(clean_row.values()):
            continue

        # Subject = first non-empty cell, with its header included for full metric context.
        subject_col = None
        subject_val = ""
        for j, (h, v) in enumerate(clean_row.items()):
            if v:
                subject_col, subject_val = h, v
                break
        if subject_col is None:
            continue

        if current_section:
            full_subject = f"{current_section} - {subject_val} ({subject_col})"
        else:
            full_subject = f"{subject_val} ({subject_col})"

        body = {h: v for h, v in clean_row.items() if h != subject_col}
        sentence = f"Table: {table_name}. In {full_subject}"
        if body:
            parts = [f"the {h} is {v}" for h, v in body.items() if v != ""]
            if parts:
                sentence += ", " + ", and ".join(parts) + "."
            else:
                sentence += "."
        else:
            sentence += "."

        docs.append(
            Document(
                page_content=sentence,
                metadata={
                    "source": source,
                    "table": table_name,
                    "row": i,
                    "type": "table_row",
                    **base_meta,
                },
            )
        )
    return docs


# Called in: multi_agent/retrieval/table_serialization.py (_rows_to_documents)
def _clean_cell(value) -> str:
    """Normalize a raw pdfplumber cell: drop newlines, collapse whitespace."""
    if value is None:
        return ""
    return " ".join(str(value).split())