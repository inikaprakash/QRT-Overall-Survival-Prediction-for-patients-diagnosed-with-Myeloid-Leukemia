import pandas as pd

SIGNALING_PATHWAYS_PROTEINS = [
    "ABL1", "BCL10", "BRAF", "CALR", "CBL", "CSF1R", "CSF3R",
    "EGFR", "ETNK1", "FLT3", "GNAS", "GNB1", "JAK2", "JAK3",
    "KIT", "KRAS", "MPL", "NF1", "NOTCH1", "NOTCH2", "NRAS",
    "PIK3CA", "PTPN11", "PTPRF", "RAC1", "ROBO1", "ROBO2",
    "RRAS", "SETBP1", "SH2B3", "SPRED2", "STAT3", "STAT5A"
]

TRANSCRIPTION_GENES = [
    "CEBPA", "CTCF", "CUX1", "ETV6", "GATA1", "GATA2", "IRF1",
    "IRF4", "IRF8", "MGA", "MYC", "NFE2", "NPM1", "PAX5",
    "RUNX1", "WT1", "ZBTB33"
]

EPIGENETIC_GENES = [
    "ARID1A", "ARID2", "ASXL1", "ASXL2", "ATRX", "BAP1", "BCOR",
    "BCORL1", "CREBBP", "DNMT3A", "DNMT3B", "EED", "EP300",
    "EZH2", "H3F3A", "IDH1", "IDH2", "JARID2", "KDM5C", "KDM6A",
    "KMT2C", "KMT2D", "MLL", "PHF6", "RBBP4", "SETD2", "SRCAP",
    "SUZ12", "TET2", "WHSC1"
]

CELL_CYCLE_GENES = [
    "CDK4", "CDKN1B", "CDKN2A", "CDKN2B", "CDKN2C", "CHEK2",
    "HIPK2", "NF2", "PPM1D", "PTEN", "RB1", "TP53"
]

SPLICING_GENES = [
    "DDX23", "LUC7L2", "PRPF40A", "PRPF8", "SF1", "SF3B1",
    "SRSF2", "U2AF1", "U2AF2", "ZRSR2"
]

COHESION_DNA_GENES = [
    "BRCC3", "DDX41", "FAM175A", "NIPBL", "RAD21", "RAD50",
    "SAMHD1", "SMC1A", "SMC3", "SMG1", "STAG1", "STAG2"
]

def molecular_features(train, test):
    # copy the data so the original tables stay unchanged
    train = train.copy()
    test = test.copy()

    # both datasets need the gene column for these features
    if "GENE" not in train.columns:
        raise ValueError("Training molecular dataframe must contain a 'GENE' column.")

    if "GENE" not in test.columns:
        raise ValueError("Test molecular dataframe must contain a 'GENE' column.")


    # make gene names consistent before matching them
    train["GENE"] = train["GENE"].astype("string").str.strip().str.upper()
    test["GENE"] = test["GENE"].astype("string").str.strip().str.upper()


    # group genes by the biological process they belong to
    gene_groups = {
        "signaling_pathways_proteins": SIGNALING_PATHWAYS_PROTEINS,
        "transcription_genes": TRANSCRIPTION_GENES,
        "epigenetic_genes": EPIGENETIC_GENES,
        "cell_cycle_genes": CELL_CYCLE_GENES,
        "splicing_genes": SPLICING_GENES,
        "cohesion_dna_genes": COHESION_DNA_GENES
    }

    # create one binary feature for each gene group
    for feature_name, genes in gene_groups.items():

        train[feature_name] = train["GENE"].isin(genes).astype(int)
        test[feature_name] = test["GENE"].isin(genes).astype(int)


    # print a quick check of how often each feature appears
    print("\nMolecular feature summary")
    print("-" * 40)

    for feature_name in gene_groups:

        train_count = train[feature_name].sum()
        test_count = test[feature_name].sum()

        print(
            f"{feature_name:<30} "
            f"train: {train_count:>5} | "
            f"test: {test_count:>5}"
        )

    return train, test


train = pd.read_csv(
    "./X_train/molecular_train.csv"
)

test = pd.read_csv(
    "./X_test/molecular_test.csv"
)

molecular_train, molecular_test = molecular_features(
    train,
    test
)

molecular_train.to_csv(
    "./X_train/molecular_train_processed.csv",
    index=False
)

molecular_test.to_csv(
    "./X_test/molecular_test_processed.csv",
    index=False
)

print("processed molecular files saved")