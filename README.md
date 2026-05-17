# Discourse Analyzer (CN religious-coded terms) — V1.5

Streamlit aplikacija kinų teksto diskurso analizei pagal žodyną (concept; term; pinyin; translation; category). Palaikomi režimai: substring, jieba (precise/search), HanLP, hybrid.

## Įdiegimas

```bash
cd discourse-analyzer-1.5-experimental
pip install -r requirements.txt
streamlit run app.py
```

---

## Rekomenduojama aplinka (venv)

Rekomenduojama naudoti virtualią aplinką (venv):

1. **Sukurti venv** (pvz. Python 3.12):  
   `py -3.12 -m venv venv312`
2. **Aktyvuoti:** PowerShell `.\venv312\Scripts\Activate.ps1` arba CMD `venv312\Scripts\activate.bat`
3. **Įdiegti:** `pip install -r requirements.txt`
4. **Paleisti:** `streamlit run app.py`

---

## HanLP (neprivaloma)

Režimui **HanLP (kinų segmentacija)** reikalinga biblioteka `hanlp` (mažas modelis COARSE_ELECTRA_SMALL_ZH).

- **Įdiegimas:** `pip install hanlp`
- **Jei importas nepavyksta:** pabandykite įdiegti backend: `pip install tensorflow` arba `pip install torch` (žr. [HanLP dokumentaciją](https://hanlp.hankcs.com/)).
- **Pirmas paleidimas:** HanLP gali atsisiųsti modelį į `HANLP_HOME` – tai normalu; reikia interneto.
- **Patikrinimas:** `python -c "import hanlp; print('HanLP OK')"`
- Jei HanLP neįdiegta arba modelis nepakrauna – pasirinkus režimą „HanLP“ bus rodomas įspėjimas su klaidos tekstu ir naudojamas **substring** fallback.
- Dokumentacija: [hanlp.hankcs.com](https://hanlp.hankcs.com/)

---

## GitHub ir Streamlit Cloud (online)

### 1. Įkelti į GitHub

```bash
git add app.py requirements.txt README.md .gitignore .streamlit/ docs/ architecture.svg
git add batch_analyze_hanlp.py batch_economist_charts.py economist_terms.txt
git add civilization_terms_UPDATED_full.csv events_CN_2017_2026.csv tanming_terms.txt
git commit -m "V1.5: Streamlit app, THULAC, batch tools, docs"
git push origin v1.5-experimental
```

Jei Streamlit Cloud naudoja `main` šaką, po sėkmingo push:

```bash
git push origin v1.5-experimental:main
```

### 2. Paleisti [Streamlit Community Cloud](https://share.streamlit.io)

1. Prisijunkite su GitHub paskyra.
2. **New app** → repozitorija `iconally/discourse-analyzer`.
3. **Branch:** `v1.5-experimental` (arba `main`, jei ten naujausia versija).
4. **Main file path:** `app.py`
5. **Deploy**

Pirmas paleidimas gali užtrukti (jieba / HanLP modeliai). Jei HanLP nepavyksta debesyje, naudokite `jieba_precise` arba `thulac` režimą UI.

Architektūros schema: `docs/architecture.md`, `docs/architecture.svg`.
