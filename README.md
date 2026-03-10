# 8-Gene Anomaly Detection (Cases vs gnomAD vs CSVS)

This repository contains code and outputs for a **gene-wise anomaly detection** experiment using an 8-gene panel:

**ELMOD1, CIB2, NPTN, MGRN1, COL9A2, KLHDC7B, GRXCR1, MARVELD2**

Because the clinical cohort contains only affected individuals, we frame the analysis as **one-class learning / anomaly detection**:
- Train an anomaly detector on **case-derived gene-level feature vectors** (disease manifold)
- Score population reference datasets as **out-of-distribution** relative to the case manifold

We evaluate against two population references:
1. **gnomAD** (multi-ancestry population reference)
2. **CSVS** (Spanish-specific population reference) for ancestry-validation

---

# File Descriptions

## Individual_data_Uniti_CPU_parallel/ (CASE_INPUT_FOLDER)

Folder containing the case cohort variant TSVs (one file per individual, e.g., `Uniti_case_*.tsv`).

Each TSV includes variant annotations such as:

- `CHROM`
- `POS`
- `REF`
- `ALT`
- `Gene (ENSG)`
- `SYMBOL`
- `Consequence`
- `CADD_PHRED`
- `AF`
- *(optionally)* `GT`

These files are filtered to the **8-gene panel** during feature extraction.

---

## gnomad_8genes_raw.tsv (GNOMAD_RAW_TSV)

Raw **gnomAD extract** for the same 8 gene regions, stored as a TSV.

This file contains **population-level variant data (no individual genotypes)**.

In our workflow it is used to compute an **aggregate gene-level feature vector per gene (real data, no simulation)** using:

- `AF`
- `CADD`
- `consequence` fields

---

## cases_gene_ml_matrix_real_8genes.csv (CASES_GENE_CSV)

Generated **feature table** with one row per **(case individual, gene)**.

Each row is a **gene-level summary of burden / severity / rarity / biallelic proxy features** used to train the **gene-wise anomaly models (Isolation Forest)** on the **case manifold**.

---

## gnomad_gene_ml_matrix_real_8genes.csv (GNOMAD_GENE_CSV)

Generated **feature table** with one row per **gene** derived from the **gnomAD raw TSV**.

This is the **population reference profile** used for evaluation against the **case-trained gene-wise anomaly detectors**.

---

## all_less_group8andControl_8_hg38.withHeader.targetGenes.tsv (CSVS reference TSV)

**CSVS (Spanish cohort) population reference TSV** restricted to the same **8 genes (GRCh38)**.

This file is treated like **gnomAD** in the real-data workflow:

- We compute **one aggregate gene-level feature vector per gene**
- Score it against **case-trained gene-wise anomaly models**

This helps validate that results are **not driven by population structure / ancestry mismatch**.
---
## Repository structure

### Input data

- `Individual_data_Uniti_CPU_parallel/`  
  Case cohort variant TSV files, one file per individual (e.g., `Uniti_case_*.tsv`).  
  Required columns: `Gene`, `SYMBOL`, `Consequence`, `CADD_PHRED`, `AF` (and optional `CHROM/POS/REF/ALT/GT`).

- `gnomad_8genes_raw.tsv`  
  Raw gnomAD TSV extract restricted to the same 8 gene regions.  
  Used to compute **aggregate gene-level features** (no individual genotypes required).

- `all_less_group8andControl_8_hg38.withHeader.targetGenes.tsv`  
  CSVS (Spanish cohort) TSV restricted to the same 8 genes (GRCh38).  
  Used as an ancestry-matched population reference for validation.

### Generated feature tables

- `cases_gene_ml_matrix_real_8genes.csv`  
  **One row per (patient, gene)** extracted from case TSVs.

- `gnomad_gene_ml_matrix_real_8genes.csv`  
  **One row per gene** extracted from gnomAD TSV.

- `csvs_gene_ml_matrix_real_8genes.csv`  
  **One row per gene** extracted from CSVS TSV.

### Outputs

- `genewise_realdata_anomaly_scores_8genes.csv`  
  Gene-wise anomaly scores and case-calibrated thresholds for gnomAD.

- `genewise_realdata_anomaly_scores_8genes_CSVS.csv`  
  Gene-wise anomaly scores and case-calibrated thresholds for CSVS.

---

## Feature engineering (gene-level)

For each gene, we compute the same feature vector for cases and population references:

**Burden**
- `variant_count`
- `rare_count` (AF < 0.01)
- `ultra_rare_count` (AF < 0.001)
- `damaging_count` (AF < 0.01 and (CADD ≥ 15 or LoF-like consequence))
- `lof_count` (keyword-based LoF: frameshift, stop gained/lost, splice donor/acceptor, start lost)

**Severity**
- `max_cadd`, `mean_cadd`, `cadd_ge20_count`

**Rarity transforms**
- `min_af`, `mean_af`
- `sum_neglog10_af`, `mean_neglog10_af`
- `cadd_weighted_rarity = sum(CADD * -log10(AF))`

**Biallelic proxy**
- `pair_count` (rare pairs with AF1*AF2 ≤ 0.001)
- `pair_score = sum(-log10(AF1*AF2))` over qualifying pairs
- `min_pair_product`

All thresholds and AF logic are kept identical across datasets.

---

## Models

### Gene-wise anomaly detection (Isolation Forest)
For each gene independently:
1. Train Isolation Forest on **case** rows for that gene (75% train split)
2. Score held-out case rows (25%) to define case anomaly distribution
3. Compute case-based thresholds:
   - 95th percentile (`case95_threshold`)
   - 99th percentile (`case99_threshold`)
4. Score the population reference gene profile (gnomAD or CSVS) and flag:
   - `flag_gt_case95`
   - `flag_gt_case99`

This yields a per-gene measure of how different the population background is from the disease manifold.

---

## How to run

### 1) Create case and gnomAD feature matrices
Run the feature generation script to produce:
- `cases_gene_ml_matrix_real_8genes.csv`
- `gnomad_gene_ml_matrix_real_8genes.csv`

### 2) Score gnomAD against the case manifold
Run the gene-wise Isolation Forest scoring script to produce:
- `genewise_realdata_anomaly_scores_8genes.csv`

### 3) Repeat for CSVS (validation)
Run the CSVS feature generation + scoring script to produce:
- `csvs_gene_ml_matrix_real_8genes.csv`
- `genewise_realdata_anomaly_scores_8genes_CSVS.csv`

---

## Notes / assumptions

- This workflow uses **real data only** (no pseudo-individual simulation) by operating at the **gene level**.
- AF must be interpreted consistently across datasets. Ideally the AF field reflects a comparable population AF definition.
- VEP consequence terms must be present for LoF keyword logic.
- Genes with too few case samples may be excluded from per-gene modelling.

## Contact
For questions or reproducibility notes, open an issue or contact the repository maintainer.
