Drop your PDF, DOCX, or TXT files here.
The multi_agent system watches this folder independently from ragbot/docs/.

On startup, multi_agent will:
  1. Compute an MD5 fingerprint of all files in this folder.
  2. If the fingerprint changed since last run, re-index into chroma_db_multi/.
  3. If unchanged, reuse the existing chroma_db_multi/ index (fast path).

ragbot/ continues to index from ragbot/docs/ — the two systems are completely separate.
