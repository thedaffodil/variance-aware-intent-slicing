# Variance-Aware Synthetic Intent Generation for Network Slicing

Code, prompts, and dataset for the paper:

> **Variance-Aware Synthetic Intent Generation for Network Slicing: A Controlled Comparison of Prompt-Level Diversity Techniques**
> *Submitted to IEEE CSCN 2026 — Track 3: Softwarization, Slicing, Automation and Network Management.*

We study how **prompt-level diversity strategies** shape LLM-generated synthetic data for **telecom network-slicing intent classification**. A fixed base prompt is held constant and only a modular *variance-enhancement* block is varied, giving six generation settings (five single strategies + one combined). Each intent is labeled with one of eight network-slice management operations grounded in the 3GPP management specifications (TS 28.531). The generated data is evaluated along **downstream learnability (separability)**, **lexical diversity**, and **embedding-based semantic diversity/separation**.

---

## Repository layout

```
.
├── dataGen_codes(thread+analyse)/   # 1) SYNTHETIC DATA GENERATION
│   ├── prompt.txt                   #    fixed base prompt (with [VARIANCE ENHANCEMENT BLOCK HERE])
│   ├── enhancements/                #    the five variance-enhancement blocks (one per strategy)
│   ├── tasks_dilara.json            #    task list (task_id, name, enhancements[])
│   ├── main2.py                     #    runner: injects a block, calls the LLM CLI, saves JSON
│   └── analyse.py                   #    label-distribution + duplicate report
│
├── generated_data/                  # 2) RAW GENERATED CORPUS (336 JSON files)
│   └── task_<id>_<technique>_<n>.json   #  one LLM call = a JSON list of {intent, slicing_operation}
│
├── loader.py                        #    shared loader (reads generated_data/)
├── balance_dataset.py               # 3) build the class-balanced dataset used in the paper
│
├── final/                           # 4) PAPER PIPELINE (balanced data + analyses + paper)
│   ├── generated_data/              #    balanced dataset (equal per setting and per label)
│   ├── loader.py                    #    load + validate + tag by technique
│   ├── separability.py              #    TF-IDF + LogReg, stratified 5-fold (learnability)
│   ├── diversity.py                 #    MATTR, Distinct-n, Self-BLEU, ...
│   ├── embed.py                     #    all-mpnet-base-v2 sentence embeddings (cached)
│   ├── semantic.py                  #    semantic diversity / class separation
│   ├── tsne_plot.py                 #    PCA(50) -> t-SNE maps
│   ├── report.py                    #    merges metrics, renders figures
│   ├── goodness_weighted.py         #    axis-weighted composite score
│   ├── output/                      #    generated CSVs + figures
│   └── writing/                     #    the LaTeX paper (main_final.tex) + figures
│
├── requirements.txt
├── LICENSE
└── README.md
```

> The eight labels: `slice_allocation`, `slice_deallocation`, `slice_list`,
> `slice_ue_quota_update`, `slice_pdu_session_quota_update`, `slice_rb_update`,
> `slice_rrc_con_update`, and `other` (in-domain but out-of-scope).

---

## The six variance settings

| Setting | Enhancement block | Inspiration |
|---|---|---|
| Taboo | Taboo opening words | Cegin et al., 2024 |
| NL+Dist | Natural language + distribution diversity | Saparina & Lapata 2024; Wang et al. 2025 |
| Scenario | Scenario-path + domain coverage | Samarinas et al. 2024; Finch & Choi 2024 |
| Hints | Task-specific hints | Saley et al. 2024 |
| Intent-Util | Intent-classification utility | Benayas et al. 2024 |
| ALL | All five blocks combined | Bao et al. 2025 |

---

## Reproduce

### 0. Environment
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

### 1. (Optional) Regenerate the corpus
`main2.py` calls an LLM through a command-line client. Point it at your own
Claude/OpenAI CLI, then:
```bash
cd "dataGen_codes(thread+analyse)"
python main2.py tasks_dilara.json --runs 40 --workers 3
```
Outputs land in `results/`. The corpus used in the paper is already provided in
`generated_data/`, so this step is optional.

### 2. Build the balanced dataset
```bash
python balance_dataset.py     # writes final/generated_data/
```

### 3. Run the analyses (paper pipeline)
```bash
cd final
python loader.py            # sanity check / load report
python separability.py      # -> output/separability_summary.csv
python diversity.py         # -> output/diversity_summary.csv
python embed.py             # builds the embedding cache (downloads all-mpnet-base-v2 once)
python semantic.py          # -> output/semantic_summary.csv + figures
python tsne_plot.py         # -> output/fig_tsne_*.png
python report.py            # -> master tables + fig_tradeoff / fig_heatmap
python goodness_weighted.py # -> output/goodness_weighted.csv
```

---

## Notes
- Semantic metrics use the `all-mpnet-base-v2` sentence-transformer; the embedding
  cache (`*.npy`) is regenerated by `embed.py` and is not tracked in git.
- Random seeds are fixed (`42`) throughout for reproducibility.
- Behind a TLS-inspecting corporate proxy, install `truststore` so that the
  Hugging Face model download uses the OS certificate store.

## Citation
```bibtex
@inproceedings{yourkey2026variance,
  title     = {Variance-Aware Synthetic Intent Generation for Network Slicing:
               A Controlled Comparison of Prompt-Level Diversity Techniques},
  author    = {Author Name(s)},
  booktitle = {Proc. IEEE Conf. on Standards for Communications and Networking (CSCN)},
  year      = {2026}
}
```

## License
Released under the MIT License — see [LICENSE](LICENSE).
