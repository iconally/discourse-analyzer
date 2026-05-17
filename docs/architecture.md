# Sinocode Discourse Analyzer — architektūra

**Pagrindinė aplikacija:** Streamlit **`app.py`** (tai, ką paleidžiate naršyklėje).  
**Papildomai (be UI):** `batch_analyze_hanlp.py` ir `batch_economist_charts.py` — tas pats `analyze_text` principas, bet masiniam korpusui ir Economist PNG/CSV išvestims.

**SVG schema** (`docs/architecture.svg`, kopija repo šaknyje `architecture.svg`): tik **Streamlit `app.py`** kelias — **be** offline batch. Visa repo su batch — žemiau Mermaid diagramose.

![Architektūros schema — tik app.py](architecture.svg)

## 1. Aukšto lygio komponentai

```mermaid
flowchart TB
  subgraph User["Vartotojas"]
    BR[Naršyklė]
  end

  subgraph UI["Pagrindinė aplikacija (UI)"]
    APP["Streamlit: app.py — entry point"]
    UP["Įkelti .txt / .docx"]
    MM["match_mode: jieba_*, substring, hybrid, thulac, hanlp"]
    APP --> UP
    APP --> MM
  end

  subgraph Libs["Pasirenkamos bibliotekos"]
    JB[jieba]
    TL[THULAC lazy]
    HL[HanLP lazy]
    DC[python-docx]
    MPL[matplotlib Agg]
  end

  subgraph Data["Duomenų failai"]
    CSV["Žodyno CSV pvz. civilization_terms_*.csv"]
    CORP["Lokalus korpusas pvz. D:\\Xi"]
  end

  subgraph Batch["Papildoma: offline CLI (ne UI)"]
    ECO["economist_terms.txt (tik batch / Economist batch)"]
    BEC["batch_economist_charts.py"]
    BAH["batch_analyze_hanlp.py"]
    ECO --> BEC
    BEC --> BAH
  end

  subgraph Out["Batch išvestis"]
    TH["term_hits_all_docs.csv"]
    SH["economist_year_share.csv"]
    PNG["economist_charts/*.png"]
    SK["skipped_*.csv, run_meta.txt"]
  end

  BR -->|HTTP| APP
  CSV --> APP
  MM --> JB
  MM --> TL
  MM --> HL
  UP --> DC

  CSV --> BEC
  CORP --> BEC
  BEC --> TH
  BEC --> SH
  BEC --> PNG
  BEC --> SK

  MM -.->|Economist metų dalies metrika kaip app.py| SH
```

**THULAC:** segmentatorius naudojamas **Streamlit** (`app.py`), kai pasirenkamas `thulac` režimas (lazy `thulac.thulac`). CLI `batch_analyze_hanlp.py` / `batch_economist_charts.py` šiuo metu `--match-mode` **neturi** `thulac` — ten galimi `jieba_precise`, `jieba_search`, `substring`, `hybrid`, `hanlp`.

## 2. Duomenų srautas (analizė)

```mermaid
flowchart LR
  T[Žali tekstai] --> A[analyze_text tekste]
  D[Žodyno DataFrame] --> A
  A --> H[Term hitai: CH term, Count, metai iš failo vardo]
  H --> U[Lentelės ir grafikai Streamlit]
  H --> C[optional: batch CSV/PNG exports]
```

## 3. Batch Economist (2017+)

```mermaid
flowchart TD
  P[rglob .txt po --docs-dir] --> F{Aplanko segmente metai žemesni nei 2017?}
  F -->|taip| S1[skip → skipped_folder_year_before_cutoff.csv]
  F -->|ne| Y{Failo vardo metai ≥ min?}
  Y -->|ne| S2[skip → skipped_before_min_year.csv]
  Y -->|taip| R[read + analyze_text]
  R --> AG[Agregacija pagal metus]
  AG --> CSV_OUT[economist_year_share.csv]
  AG --> CHART[PNG mini diagramos]
```

## 4. Statinis paveikslėlis (SVG, be batch)

- [`docs/architecture.svg`](architecture.svg) ir [`architecture.svg`](../architecture.svg) — **tik** pagrindinė aplikacija (`app.py`).
- Trečias blokas vadinasi **Term list (CSV)** (ne „Lexicon“): tai aiškiau nei angl. *lexicon* („žodynas“, sąrašas leksikų). Seniau diagramoje buvo bendrinis angliškas žodis be konteksto.
- SVG **nemini** Economist pavadinimo (batch / atskiros ataskaitos lieka Mermaid skyriuose).
- Pilnas vaizdas su CLI batch: **1 skyrius** (Mermaid) ir šis `.md` failas.

## 5. Kaip peržiūrėti Mermaid

- GitHub / GitLab: šis `.md` failas dažnai automatiškai piešia diagramas.
- [mermaid.live](https://mermaid.live): įklijuokite `flowchart` bloką.
