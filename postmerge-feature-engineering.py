import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis


TARGET_COLS = ["OS_YEARS", "OS_STATUS"]

# these are the gene groups used for the extra scores
GENE_CATEGORIES = {
        "SIGNALING_PATHWAYS_PROTEINS" : [
        "ABL1", "BCL10", "BRAF", "CALR", "CBL", "CSF1R", "CSF3R",
        "EGFR", "ETNK1", "FLT3", "GNAS", "GNB1", "JAK2", "JAK3",
        "KIT", "KRAS", "MPL", "NF1", "NOTCH1", "NOTCH2", "NRAS",
        "PIK3CA", "PTPN11", "PTPRF", "RAC1", "ROBO1", "ROBO2",
        "RRAS", "SETBP1", "SH2B3", "SPRED2", "STAT3", "STAT5A"
    ],
    "TRANSCRIPTION_GENES" : [
        "CEBPA", "CTCF", "CUX1", "ETV6", "GATA1", "GATA2", "IRF1",
        "IRF4", "IRF8", "MGA", "MYC", "NFE2", "NPM1", "PAX5",
        "RUNX1", "WT1", "ZBTB33"
    ],

    "EPIGENETIC_GENES" : [
        "ARID1A", "ARID2", "ASXL1", "ASXL2", "ATRX", "BAP1", "BCOR",
        "BCORL1", "CREBBP", "DNMT3A", "DNMT3B", "EED", "EP300",
        "EZH2", "H3F3A", "IDH1", "IDH2", "JARID2", "KDM5C", "KDM6A",
        "KMT2C", "KMT2D", "MLL", "PHF6", "RBBP4", "SETD2", "SRCAP",
        "SUZ12", "TET2", "WHSC1"
    ],

    "CELL_CYCLE_GENES" : [
        "CDK4", "CDKN1B", "CDKN2A", "CDKN2B", "CDKN2C", "CHEK2",
        "HIPK2", "NF2", "PPM1D", "PTEN", "RB1", "TP53"
    ],

    "SPLICING_GENES" : [
        "DDX23", "LUC7L2", "PRPF40A", "PRPF8", "SF1", "SF3B1",
        "SRSF2", "U2AF1", "U2AF2", "ZRSR2"
    ],

    "COHESION_DNA_GENES" : [
        "BRCC3", "DDX41", "FAM175A", "NIPBL", "RAD21", "RAD50",
        "SAMHD1", "SMC1A", "SMC3", "SMG1", "STAG1", "STAG2"
    ],
}

AGGREGATED_CATEGORY_COLUMNS = {
    "SIGNALING_PATHWAYS_PROTEINS": "signaling_pathways_proteins_max",
    "TRANSCRIPTION_GENES": "transcription_genes_max",
    "EPIGENETIC_GENES": "epigenetic_genes_max",
    "CELL_CYCLE_GENES": "cell_cycle_genes_max",
    "SPLICING_GENES": "splicing_genes_max",
    "COHESION_DNA_GENES": "cohesion_dna_genes_max",
}


def make_survival_target(df):
    # put the survival columns into the format cox needs
    return np.array(
        [
            (bool(event), float(time))
            for event, time in zip(
                df["OS_STATUS"],
                df["OS_YEARS"]
            )
        ],
        dtype=[
            ("event", "?"),
            ("time", "<f8")
        ]
    )


def split_data(df, random_state=42):
    # clean the targets before making the patient split
    df = df.copy()

    if "OS_STATUS" not in df.columns:
        raise ValueError("OS_STATUS not found in dataframe.")

    df["OS_YEARS"] = pd.to_numeric(
        df["OS_YEARS"],
        errors="coerce"
    )
    df["OS_STATUS"] = pd.to_numeric(
        df["OS_STATUS"],
        errors="coerce"
    )
    df = df.dropna(
        subset=["OS_YEARS", "OS_STATUS"]
    ).reset_index(drop=True)

    if df["OS_STATUS"].nunique() < 2:
        raise ValueError(
            "OS_STATUS must contain both event classes after cleaning."
        )

    # first keep 60% for training
    train_df, temp_df = train_test_split(
        df,
        test_size=0.40,
        random_state=random_state,
        stratify=df["OS_STATUS"]
    )

    # split the remaining 40% into validation and internal test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df["OS_STATUS"]
    )

    # make sure no patient appears in more than one split
    train_ids = set(train_df["ID"])
    val_ids = set(val_df["ID"])
    test_ids = set(test_df["ID"])

    if (
        train_ids & val_ids
        or train_ids & test_ids
        or val_ids & test_ids
    ):
        raise ValueError("patient IDs overlap between splits.")

    # make the indexes neat again
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def get_gene_columns(df, gene_categories):
    # only use gene features that are actually in the merged data
    available_genes = []

    for genes in gene_categories.values():
        for gene in genes:
            if gene in df.columns:
                available_genes.append(gene)

    for column in AGGREGATED_CATEGORY_COLUMNS.values():
        if column in df.columns:
            available_genes.append(column)

    # remove duplicates but keep the original order
    return list(dict.fromkeys(available_genes))


def estimate_gene_effects(
    train_df,
    gene_categories,
    min_mutations=10,
    alpha=1.0
):
    # learn the effects from training data only
    y_train = make_survival_target(train_df)

    gene_effects = {}

    gene_columns = get_gene_columns(
        train_df,
        gene_categories
    )

    for gene in gene_columns:

        # make sure the feature is numeric
        gene_values = pd.to_numeric(
            train_df[gene],
            errors="coerce"
        ).fillna(0)

        # skip features that are too rare
        n_mutated = int(gene_values.sum())

        if n_mutated < min_mutations:
            continue

        # cox needs both zero and one values
        if gene_values.nunique() < 2:
            continue

        X_gene = pd.DataFrame({
            gene: gene_values.astype(float)
        })

        try:

            cox = CoxPHSurvivalAnalysis(
                alpha=alpha
            )

            cox.fit(
                X_gene,
                y_train
            )

            gene_effects[gene] = float(
                cox.coef_[0]
            )

        except Exception:
            # ignore features that fail to fit
            continue

    return gene_effects


def add_target_specific_scores(
    df,
    gene_effects,
    gene_categories
):
    # apply the training effects to each dataset
    df = df.copy()

    for category, genes in gene_categories.items():

        score = np.zeros(len(df))

        feature_names = list(genes)
        aggregate_column = AGGREGATED_CATEGORY_COLUMNS.get(category)

        if aggregate_column is not None:
            feature_names.append(aggregate_column)

        for gene in feature_names:

            # skip genes without a learned effect
            if gene not in gene_effects:
                continue

            # skip columns that are not available here
            if gene not in df.columns:
                continue

            gene_values = pd.to_numeric(
                df[gene],
                errors="coerce"
            ).fillna(0)

            score += (
                gene_values.astype(float)
                * gene_effects[gene]
            )

        df[f"{category}_survival_score"] = score

    return df


def create_feature_splits(
    merged_df,
    gene_categories=GENE_CATEGORIES,
    random_state=42,
    min_mutations=10,
    alpha=1.0
):
        # split first so validation stays separate
    train_df, val_df, test_df = split_data(
        merged_df,
        random_state=random_state
    )

    # learn effects only from the training part
    gene_effects = estimate_gene_effects(
        train_df=train_df,
        gene_categories=gene_categories,
        min_mutations=min_mutations,
        alpha=alpha
    )

    # use the same learned effects everywhere
    train_df = add_target_specific_scores(
        train_df,
        gene_effects,
        gene_categories
    )

    val_df = add_target_specific_scores(
        val_df,
        gene_effects,
        gene_categories
    )

    test_df = add_target_specific_scores(
        test_df,
        gene_effects,
        gene_categories
    )

    return (
        train_df,
        val_df,
        test_df,
        gene_effects
    )


def gene_effect_table(gene_effects, gene_categories):
    # make the learned effects easier to read
    rows = []

    for category, genes in gene_categories.items():

        feature_names = list(genes)
        aggregate_column = AGGREGATED_CATEGORY_COLUMNS.get(category)

        if aggregate_column is not None:
            feature_names.append(aggregate_column)

        for gene in feature_names:

            if gene not in gene_effects:
                continue

            beta = gene_effects[gene]

            rows.append({
                "category": category,
                "gene": gene,
                "cox_beta": beta,
                "hazard_ratio": np.exp(beta)
            })

    effects = pd.DataFrame(rows)

    if not effects.empty:
        effects = effects.sort_values(
            ["category", "cox_beta"]
        ).reset_index(drop=True)

    return effects


if __name__ == "__main__":

    merged_df = pd.read_csv(
        "./X_train/train_merged.csv"
    )

    (
        train_df,
        val_df,
        test_df,
        gene_effects
    ) = create_feature_splits(
        merged_df,
        gene_categories=GENE_CATEGORIES,
        random_state=42,
        min_mutations=10,
        alpha=1.0
    )

    # check the sizes of the three groups
    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    # check that the event rates are similar
    print("\nEvent proportions:")

    print(
        "Train:",
        train_df["OS_STATUS"].mean()
    )

    print(
        "Validation:",
        val_df["OS_STATUS"].mean()
    )

    print(
        "Test:",
        test_df["OS_STATUS"].mean()
    )

    # show the effects learned from training
    effects = gene_effect_table(
        gene_effects,
        GENE_CATEGORIES
    )

    print("\nGene-specific Cox effects:")
    print(effects.to_string(index=False))

    # list the new score columns
    score_cols = [
        col
        for col in train_df.columns
        if col.endswith("_survival_score")
    ]

    print("\nTarget-specific features:")
    print(score_cols)

    print("\nTraining preview:")
    print(
        train_df[
            ["ID"] + score_cols
        ].head()
    )

    train_df.to_csv(
        "./X_train/train_60_processed.csv",
        index=False
    )

    val_df.to_csv(
        "./X_train/validation_20_processed.csv",
        index=False
    )

    test_df.to_csv(
        "./X_train/test_20_processed.csv",
        index=False
    )

    print("saved the 60/20/20 processed files")

