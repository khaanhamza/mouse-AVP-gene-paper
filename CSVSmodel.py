import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# =========================================================
# CONFIG
# =========================================================

CASE_INPUT_FOLDER = "Individual_data_Uniti_CPU_parallel"

CSVS_TSV = "all_less_group8andControl_8_hg38.withHeader.targetGenes.tsv"

CASES_OUT = "cases_gene_ml_matrix_real_8genes.csv"
CSVS_OUT  = "csvs_gene_ml_matrix_real_8genes.csv"

OUT_SCORES = "genewise_realdata_anomaly_scores_8genes_CSVS.csv"

TARGET_SYMBOLS = [
    "ELMOD1",
    "CIB2",
    "NPTN",
    "MGRN1",
    "COL9A2",
    "KLHDC7B",
    "GRXCR1",
    "MARVELD2",
]

TARGET_ENSEMBL_IDS = {
    "ENSG00000110675",  # ELMOD1
    "ENSG00000136425",  # CIB2
    "ENSG00000156642",  # NPTN
    "ENSG00000102858",  # MGRN1
    "ENSG00000049089",  # COL9A2
    "ENSG00000130487",  # KLHDC7B
    "ENSG00000215203",  # GRXCR1
    "ENSG00000152939",  # MARVELD2
}

# thresholds/logic EXACTLY same as your latest isolation forest method
RARE_AF_THRESHOLD = 0.01
ULTRA_RARE_AF_THRESHOLD = 0.001
DAMAGING_CADD_THRESHOLD = 15.0
PAIR_PRODUCT_THRESHOLD = 0.001

LOF_KEYWORDS = [
    "frameshift",
    "stop_gained",
    "stop_lost",
    "splice_donor",
    "splice_acceptor",
    "start_lost",
]

RANDOM_SEED = 42


# =========================================================
# HELPERS
# =========================================================

def normalize_gene_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.split(".").str[0]

def is_lof_consequence(cons: str) -> bool:
    if pd.isna(cons):
        return False
    c = str(cons).lower()
    return any(k in c for k in LOF_KEYWORDS)

def safe_neglog10(x):
    if pd.isna(x):
        return np.nan
    x = float(x)
    if x <= 0:
        return np.nan
    return -np.log10(x)

def pair_metrics_from_af(af_values, threshold=0.001):
    vals = [float(x) for x in af_values if pd.notna(x)]
    n = len(vals)
    if n < 2:
        return 0, 0.0, np.nan

    count = 0
    pair_score = 0.0
    min_pair_product = np.nan

    for i in range(n):
        for j in range(i + 1, n):
            prod = vals[i] * vals[j]
            if prod <= threshold:
                count += 1
                if prod > 0:
                    pair_score += -np.log10(prod)
                if pd.isna(min_pair_product) or prod < min_pair_product:
                    min_pair_product = prod

    return count, pair_score, min_pair_product

def compute_gene_features(g: pd.DataFrame, unit_id: str, gene_symbol: str) -> dict:
    g = g.copy()

    g["AF"] = pd.to_numeric(g["AF"], errors="coerce")
    g["CADD_PHRED"] = pd.to_numeric(g["CADD_PHRED"], errors="coerce")
    g["neglog10_af"] = g["AF"].apply(safe_neglog10)
    g["is_lof"] = g["Consequence"].apply(is_lof_consequence).astype(int)

    rare = g[g["AF"].notna() & (g["AF"] < RARE_AF_THRESHOLD)].copy()
    ultra_rare = g[g["AF"].notna() & (g["AF"] < ULTRA_RARE_AF_THRESHOLD)].copy()

    damaging = g[
        g["AF"].notna() &
        (g["AF"] < RARE_AF_THRESHOLD) &
        (
            (g["CADD_PHRED"].notna() & (g["CADD_PHRED"] >= DAMAGING_CADD_THRESHOLD)) |
            (g["is_lof"] == 1)
        )
    ].copy()

    pair_count, pair_score, min_pair_product = pair_metrics_from_af(
        rare["AF"].tolist(),
        threshold=PAIR_PRODUCT_THRESHOLD
    )

    cadd_weighted_rarity = (
        (g["CADD_PHRED"].fillna(0) * g["neglog10_af"].fillna(0)).sum()
        if len(g) else 0.0
    )

    row = {
        "unit_id": unit_id,
        "gene": gene_symbol,

        "variant_count": int(len(g)),
        "rare_count": int(len(rare)),
        "ultra_rare_count": int(len(ultra_rare)),
        "damaging_count": int(len(damaging)),
        "lof_count": int(g["is_lof"].sum()),

        "max_cadd": float(g["CADD_PHRED"].max()) if g["CADD_PHRED"].notna().any() else 0.0,
        "mean_cadd": float(g["CADD_PHRED"].mean()) if g["CADD_PHRED"].notna().any() else 0.0,
        "cadd_ge20_count": int((g["CADD_PHRED"].fillna(0) >= 20).sum()),

        "min_af": float(g["AF"].min()) if g["AF"].notna().any() else 1.0,
        "mean_af": float(g["AF"].mean()) if g["AF"].notna().any() else 1.0,
        "sum_neglog10_af": float(g["neglog10_af"].fillna(0).sum()),
        "mean_neglog10_af": float(g["neglog10_af"].fillna(0).mean()) if len(g) else 0.0,
        "cadd_weighted_rarity": float(cadd_weighted_rarity),

        "pair_count": int(pair_count),
        "pair_score": float(pair_score),
        "min_pair_product": float(min_pair_product) if pd.notna(min_pair_product) else 1.0,
    }
    return row


# -------------------------
# CSVS: robust column mapping
# -------------------------

def standardize_csvs_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to produce canonical columns:
      Gene, SYMBOL, Consequence, CADD_PHRED, AF
    from typical VEP TSV outputs.
    """
    df = df.copy()
    cols = df.columns.tolist()

    def pick(*cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    # Common VEP TSV column possibilities
    col_gene = pick("Gene", "gene", "Gene_ID", "Ensembl_Gene", "Ensembl_gene_id", "ENSG", "EnsemblGeneID")
    col_sym  = pick("SYMBOL", "Symbol", "gene_symbol", "Gene_Name", "HGNC", "hgnc_symbol")
    col_cons = pick("Consequence", "consequence", "Consequence_terms", "Most_severe_consequence", "most_severe_consequence")
    col_cadd = pick("CADD_PHRED", "cadd_phred", "CADD", "CADD_PHRED_SCORE", "CADD_PHRED_score")
    col_af   = pick("AF", "af", "gnomad_AF", "gnomAD_AF", "gnomad_af", "Allele_Frequency", "allele_frequency")

    # If CSVS file came from split-vep, it might already have these exact names.
    rename_map = {}
    if col_gene: rename_map[col_gene] = "Gene"
    if col_sym:  rename_map[col_sym]  = "SYMBOL"
    if col_cons: rename_map[col_cons] = "Consequence"
    if col_cadd: rename_map[col_cadd] = "CADD_PHRED"
    if col_af:   rename_map[col_af]   = "AF"

    df = df.rename(columns=rename_map)

    # Some VEP TSVs store symbol in "SYMBOL" but gene ID in another column, or vice versa.
    # Ensure these exist after rename.
    required = {"Gene", "SYMBOL", "Consequence", "CADD_PHRED", "AF"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("CSVS columns available (first 60):", cols[:60])
        raise ValueError(f"CSVS TSV missing required columns after mapping: {missing}")

    # Normalize types
    df["Gene"] = normalize_gene_id(df["Gene"])
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()

    return df


# =========================================================
# BUILD CASE MATRIX: one row per (patient, gene)
# =========================================================

case_rows = []
case_files = sorted(glob.glob(os.path.join(CASE_INPUT_FOLDER, "Uniti_case_*.tsv")))

for fp in tqdm(case_files, desc="Cases -> gene matrix", unit="file"):
    try:
        df = pd.read_csv(fp, sep="\t", low_memory=False)
    except Exception as e:
        print(f"❌ Failed to read {os.path.basename(fp)}: {e}")
        continue

    required = {"Gene", "SYMBOL", "Consequence", "CADD_PHRED", "AF"}
    if not required.issubset(df.columns):
        print(f"⚠️ Skipping {os.path.basename(fp)}: missing columns {sorted(required - set(df.columns))}")
        continue

    df["Gene"] = normalize_gene_id(df["Gene"])
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()

    df = df[
        df["Gene"].isin(TARGET_ENSEMBL_IDS) |
        df["SYMBOL"].isin(TARGET_SYMBOLS)
    ].copy()

    if df.empty:
        continue

    patient_id = os.path.basename(fp)

    for gene in TARGET_SYMBOLS:
        g = df[df["SYMBOL"] == gene].copy()
        if g.empty:
            continue
        case_rows.append(compute_gene_features(g, patient_id, gene))

cases_gene = pd.DataFrame(case_rows)
cases_gene.to_csv(CASES_OUT, index=False)

print(f"\n✅ Saved cases gene matrix: {CASES_OUT}")
print(f"Rows: {len(cases_gene)}")
print(cases_gene.head(3))


# =========================================================
# BUILD CSVS MATRIX: one row per gene (real aggregate)
# =========================================================

csvs_raw = pd.read_csv(CSVS_TSV, sep="\t", low_memory=False)
csvs = standardize_csvs_columns(csvs_raw)

csvs = csvs[
    csvs["Gene"].isin(TARGET_ENSEMBL_IDS) |
    csvs["SYMBOL"].isin(TARGET_SYMBOLS)
].copy()

csvs["AF"] = pd.to_numeric(csvs["AF"], errors="coerce")
csvs["CADD_PHRED"] = pd.to_numeric(csvs["CADD_PHRED"], errors="coerce")

csvs = csvs.dropna(subset=["SYMBOL", "Gene", "Consequence", "AF", "CADD_PHRED"])

csvs_rows = []
for gene in TARGET_SYMBOLS:
    gg = csvs[csvs["SYMBOL"] == gene].copy()
    if gg.empty:
        continue
    csvs_rows.append(compute_gene_features(gg, "csvs", gene))

csvs_gene = pd.DataFrame(csvs_rows)
csvs_gene.to_csv(CSVS_OUT, index=False)

print(f"\n✅ Saved CSVS gene matrix: {CSVS_OUT}")
print(f"Rows: {len(csvs_gene)}")
print(csvs_gene.head(8))


# =========================================================
# GENE-WISE ISOLATION FOREST: cases-trained -> score CSVS
# =========================================================

cases = pd.read_csv(CASES_OUT)
csvs_ref = pd.read_csv(CSVS_OUT)

feature_cols = [c for c in cases.columns if c not in ["unit_id", "gene"]]

results = []

for gene in sorted(set(cases["gene"]) & set(csvs_ref["gene"])):
    case_g = cases[cases["gene"] == gene].copy()
    csvs_g = csvs_ref[csvs_ref["gene"] == gene].copy()

    if len(case_g) < 10 or len(csvs_g) != 1:
        print(f"⚠️ Skipping {gene}: case_n={len(case_g)}, csvs_n={len(csvs_g)}")
        continue

    X_cases = case_g[feature_cols].copy()
    X_csvs = csvs_g[feature_cols].copy()

    X_train, X_ref = train_test_split(
        X_cases,
        test_size=0.25,
        random_state=RANDOM_SEED
    )

    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imp)

    X_ref_imp = imputer.transform(X_ref)
    X_ref_scaled = scaler.transform(X_ref_imp)

    X_csvs_imp = imputer.transform(X_csvs)
    X_csvs_scaled = scaler.transform(X_csvs_imp)

    iso = IsolationForest(
        n_estimators=800,
        contamination="auto",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    iso.fit(X_train_scaled)

    case_ref_normality = iso.decision_function(X_ref_scaled)
    csvs_normality = iso.decision_function(X_csvs_scaled)[0]

    case_ref_anomaly = -case_ref_normality
    csvs_anomaly = -csvs_normality

    thr95 = float(np.quantile(case_ref_anomaly, 0.95))
    thr99 = float(np.quantile(case_ref_anomaly, 0.99))

    results.append({
        "gene": gene,
        "case_n": len(case_g),
        "case_ref_mean_anomaly": float(case_ref_anomaly.mean()),
        "case95_threshold": thr95,
        "case99_threshold": thr99,
        "csvs_anomaly_score": float(csvs_anomaly),
        "delta_vs_case_mean": float(csvs_anomaly - case_ref_anomaly.mean()),
        "flag_gt_case95": int(csvs_anomaly > thr95),
        "flag_gt_case99": int(csvs_anomaly > thr99),
    })

out = pd.DataFrame(results).sort_values("delta_vs_case_mean", ascending=False)
out.to_csv(OUT_SCORES, index=False)

print("\n✅ Saved:", OUT_SCORES)
print(out.to_string(index=False))
