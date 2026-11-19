# M&A + Insurance Mini Project

| File | What it is |
|---|---|
| `Part_A_Insurance_Claim_Prediction.ipynb` | Part A (30%). Fully executed: outputs and charts are saved inside. |
| `Part_B_MA_RAG_Assistant.ipynb` | Part B (70%). LangChain + FAISS RAG assistant, with the written answers. Run it to see outputs. |
| `rag_app.py` | The same RAG code as a command-line tool. |
| `insurance2__1_.csv`, `M_A_Due_Diligence_Playbook.pdf` | The provided inputs. |
| `figs_a/` | PNG copies of the Part A charts. |
| `requirements.txt` | `pip install -r requirements.txt` |

## Running it
1. `pip install -r requirements.txt`
2. Part A: open the notebook and run all cells (it already contains results).
3. Part B: optionally `export OPENAI_API_KEY=...` (without it you get retrieval-only mode with a free local embedding model), then run the notebook, or `python rag_app.py "What are earn-outs?"`.

## Things to know
- **Part A was executed; Part B was not.** LangChain/FAISS could not be installed where I built this (no internet), so the Part B code is untested end-to-end. I checked that it compiles, and ran a keyword-based stand-in for the chunk-size experiment. Expect to fix a small import or version issue on first run.
- The playbook PDF is only 2 pages and has no TargetCo case study, NDA lifecycle or integration section, so those questions should return "not available".
- Part A scores are very high (about 98% accuracy), which suggests the dataset is partly synthetic. See the caveat inside the notebook.
