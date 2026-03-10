import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================

CASE_INPUT_FOLDER = "Individual_data_Uniti_CPU_parallel"
GNOMAD_RAW_TSV = "gnomad_8genes_raw.tsv"

CASES_OUT = "cases_gene_ml_matrix_real_8genes.csv"
GNOMAD_OUT = "gnomad_gene_ml_matrix_real_8genes.csv"

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

# gnomAD raw file layout (no header)
GNOMAD_COLS = ["CHROM", "POS", "REF", "ALT", "AF", "CADD_PHRED", "FILTER", "AC", "AN", "CSQ"]


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

def pick_best_vep_record(csq: str):
    """
    Return (Consequence, SYMBOL, Gene) from best transcript record.
    Prefer records with Ensembl gene IDs.
    """
    if pd.isna(csq) or not str(csq).strip():
        return (np.nan, np.nan, np.nan)

    records = str(csq).split(",")
    best = None

    for r in records:
        parts = r.split("|")
        if len(parts) < 5:
            continue
        allele, consequence, impact, symbol, gene = parts[:5]

        if str(gene).startswith("ENSG"):
            return (consequence, symbol, gene)

        if best is None and symbol:
            best = (consequence, symbol, gene)

    return best if best is not None else (np.nan, np.nan, np.nan)

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
# BUILD GNOMAD MATRIX: one row per gene (real aggregate)
# =========================================================

g = pd.read_csv(
    GNOMAD_RAW_TSV,
    sep="\t",
    header=None,
    names=GNOMAD_COLS,
    low_memory=False
)

g["AF"] = pd.to_numeric(g["AF"], errors="coerce")
g["CADD_PHRED"] = pd.to_numeric(g["CADD_PHRED"], errors="coerce")

g[["Consequence", "SYMBOL", "Gene"]] = g["CSQ"].apply(lambda x: pd.Series(pick_best_vep_record(x)))
g["Gene"] = normalize_gene_id(g["Gene"])
g["SYMBOL"] = g["SYMBOL"].astype(str).str.strip()

g = g[
    g["Gene"].isin(TARGET_ENSEMBL_IDS) |
    g["SYMBOL"].isin(TARGET_SYMBOLS)
].copy()

g = g.dropna(subset=["SYMBOL", "Gene", "Consequence", "AF", "CADD_PHRED"])

gnomad_rows = []
for gene in TARGET_SYMBOLS:
    gg = g[g["SYMBOL"] == gene].copy()
    if gg.empty:
        continue
    gnomad_rows.append(compute_gene_features(gg, "gnomad", gene))

gnomad_gene = pd.DataFrame(gnomad_rows)
gnomad_gene.to_csv(GNOMAD_OUT, index=False)

print(f"\n✅ Saved gnomAD gene matrix: {GNOMAD_OUT}")
print(f"Rows: {len(gnomad_gene)}")
print(gnomad_gene.head(8))


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

CASES_GENE_CSV = "cases_gene_ml_matrix_real_8genes.csv"
GNOMAD_GENE_CSV = "gnomad_gene_ml_matrix_real_8genes.csv"

OUT_CSV = "genewise_realdata_anomaly_scores_8genes.csv"

RANDOM_SEED = 42

cases = pd.read_csv(CASES_GENE_CSV)
gnomad = pd.read_csv(GNOMAD_GENE_CSV)

feature_cols = [c for c in cases.columns if c not in ["unit_id", "gene"]]

results = []

for gene in sorted(set(cases["gene"]) & set(gnomad["gene"])):
    case_g = cases[cases["gene"] == gene].copy()
    gnomad_g = gnomad[gnomad["gene"] == gene].copy()

    if len(case_g) < 10 or len(gnomad_g) != 1:
        print(f"⚠️ Skipping {gene}: case_n={len(case_g)}, gnomad_n={len(gnomad_g)}")
        continue

    X_cases = case_g[feature_cols].copy()
    X_gnomad = gnomad_g[feature_cols].copy()

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

    X_gnomad_imp = imputer.transform(X_gnomad)
    X_gnomad_scaled = scaler.transform(X_gnomad_imp)

    iso = IsolationForest(
        n_estimators=800,
        contamination="auto",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    iso.fit(X_train_scaled)

    case_ref_normality = iso.decision_function(X_ref_scaled)
    gnomad_normality = iso.decision_function(X_gnomad_scaled)[0]

    case_ref_anomaly = -case_ref_normality
    gnomad_anomaly = -gnomad_normality

    thr95 = float(np.quantile(case_ref_anomaly, 0.95))
    thr99 = float(np.quantile(case_ref_anomaly, 0.99))

    results.append({
        "gene": gene,
        "case_n": len(case_g),
        "case_ref_mean_anomaly": float(case_ref_anomaly.mean()),
        "case95_threshold": thr95,
        "case99_threshold": thr99,
        "gnomad_anomaly_score": float(gnomad_anomaly),
        "delta_vs_case_mean": float(gnomad_anomaly - case_ref_anomaly.mean()),
        "flag_gt_case95": int(gnomad_anomaly > thr95),
        "flag_gt_case99": int(gnomad_anomaly > thr99),
    })

out = pd.DataFrame(results).sort_values("delta_vs_case_mean", ascending=False)
out.to_csv(OUT_CSV, index=False)

print("\n✅ Saved:", OUT_CSV)
print(out.to_string(index=False))
