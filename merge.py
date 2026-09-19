import pandas as pd
import numpy as np


def aggregate_molecular_data(molecular_df):

    df = molecular_df.copy()

    if "ID" not in df.columns:
        raise ValueError("Molecular dataframe must contain an 'ID' column.")

    pathway_cols = [
        "signaling_pathways_proteins",
        "transcription_genes",
        "epigenetic_genes",
        "cell_cycle_genes",
        "splicing_genes",
        "cohesion_dna_genes",
    ]

    pathway_cols = [
        col for col in pathway_cols
        if col in df.columns
    ]

    if "VAF" in df.columns:
        df["VAF"] = pd.to_numeric(df["VAF"], errors="coerce")

    if "DEPTH" in df.columns:
        df["DEPTH"] = pd.to_numeric(df["DEPTH"], errors="coerce")

    df["_mutation"] = 1

    agg_dict = {
        "_mutation": "sum",
    }

    if "VAF" in df.columns:
        agg_dict["VAF"] = ["mean", "median", "max", "min"]

    if "DEPTH" in df.columns:
        agg_dict["DEPTH"] = ["mean", "median", "max"]

    for col in pathway_cols:
        agg_dict[col] = "max"

    molecular = df.groupby("ID").agg(agg_dict)

    molecular.columns = [
        "_".join(col).strip("_")
        if isinstance(col, tuple)
        else col
        for col in molecular.columns
    ]

    molecular = molecular.reset_index()

    molecular = molecular.rename(
        columns={
            "_mutation_sum": "mutation_count",
            "mutation_sum": "mutation_count",
            "VAF_mean": "vaf_mean",
            "VAF_median": "vaf_median",
            "VAF_max": "vaf_max",
            "VAF_min": "vaf_min",
            "DEPTH_mean": "depth_mean",
            "DEPTH_median": "depth_median",
            "DEPTH_max": "depth_max",
        }
    )

    return molecular


def merge_train_data(clinical_df, molecular_df, target_df):

    clinical = clinical_df.copy()
    molecular = molecular_df.copy()
    target = target_df.copy()

    for name, df in [
        ("clinical", clinical),
        ("molecular", molecular),
        ("target", target),
    ]:
        if "ID" not in df.columns:
            raise ValueError(
                f"{name.capitalize()} dataframe must contain an 'ID' column."
            )

    if not clinical["ID"].is_unique:
        raise ValueError(
            "Clinical dataframe contains duplicate IDs."
        )

    if not target["ID"].is_unique:
        raise ValueError(
            "Target dataframe contains duplicate IDs."
        )

    molecular = aggregate_molecular_data(molecular)

    if not molecular["ID"].is_unique:
        raise ValueError(
            "Molecular aggregation failed: duplicate IDs remain."
        )

    clinical_ids = set(clinical["ID"])
    target_ids = set(target["ID"])

    missing_clinical = target_ids - clinical_ids

    if missing_clinical:
        raise ValueError(
            f"{len(missing_clinical)} target IDs are missing "
            "from clinical data."
        )

    train = clinical.merge(
        molecular,
        on="ID",
        how="left",
        validate="one_to_one"
    )

    molecular_feature_cols = [
        "mutation_count",
        "vaf_mean",
        "vaf_median",
        "vaf_max",
        "vaf_min",
        "depth_mean",
        "depth_median",
        "depth_max",
        "signaling_pathways_proteins_max",
        "transcription_genes_max",
        "epigenetic_genes_max",
        "cell_cycle_genes_max",
        "splicing_genes_max",
        "cohesion_dna_genes_max",
    ]

    molecular_feature_cols = [
        col for col in molecular_feature_cols
        if col in train.columns
    ]

    train[molecular_feature_cols] = train[molecular_feature_cols].fillna(0)

    train = train.merge(
        target,
        on="ID",
        how="inner",
        validate="one_to_one"
    )

    if len(train) != len(target):
        raise ValueError(
            f"Expected {len(target)} patients but obtained {len(train)}."
        )

    if not train["ID"].is_unique:
        raise ValueError(
            "Final merged dataframe contains duplicate patient IDs."
        )

    return train


if __name__ == "__main__":

    clinical_train = pd.read_csv(
        "./X_train/clinical_train_processed.csv"
    )

    molecular_train = pd.read_csv(
        "./X_train/molecular_train_processed.csv"
    )

    target_train = pd.read_csv(
        "./X_target/target_train.csv"
    )

    train = merge_train_data(
        clinical_train,
        molecular_train,
        target_train
    )

    print("Merged training shape:", train.shape)

    print("\nNumber of unique patients:")
    print(train["ID"].nunique())

    print("\nColumns:")
    print(train.columns.tolist())

    print("\nTarget:")
    print(train[["OS_YEARS", "OS_STATUS"]].head())

    print("\nMissing values:")
    print(train.isna().sum())

    print("\nMolecular feature preview:")

    molecular_features = [
        col for col in train.columns
        if col in [
            "mutation_count",
            "vaf_mean",
            "vaf_median",
            "vaf_max",
            "vaf_min",
            "depth_mean",
            "depth_median",
            "depth_max",
        ]
        or col in [
            "signaling_pathways_proteins_max",
            "transcription_genes_max",
            "epigenetic_genes_max",
            "cell_cycle_genes_max",
            "splicing_genes_max",
            "cohesion_dna_genes_max",
        ]
    ]

    print(train[["ID"] + molecular_features].head())

    train.to_csv(
        "./X_train/train_merged.csv",
        index=False
    )

    print("\nSaved: ./X_train/train_merged.csv")