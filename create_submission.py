import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis


TRAIN_PATH = "./X_train/train_merged.csv"
CLINICAL_TEST_PATH = "./X_test/clinical_test_processed.csv"
MOLECULAR_TEST_PATH = "./X_test/molecular_test_processed.csv"
TARGET_PATH = "./X_target/target_train.csv"
OUTPUT_PATH = "./submission_elastic_net_cox.csv"


PATHWAY_COLUMNS = [
    "signaling_pathways_proteins",
    "transcription_genes",
    "epigenetic_genes",
    "cell_cycle_genes",
    "splicing_genes",
    "cohesion_dna_genes",
]


def aggregate_molecular_data(dataframe):
    dataframe = dataframe.copy()
    dataframe["_mutation"] = 1

    aggregation = {"_mutation": "sum"}
    if "VAF" in dataframe.columns:
        dataframe["VAF"] = pd.to_numeric(dataframe["VAF"], errors="coerce")
        aggregation["VAF"] = ["mean", "median", "max", "min"]
    if "DEPTH" in dataframe.columns:
        dataframe["DEPTH"] = pd.to_numeric(dataframe["DEPTH"], errors="coerce")
        aggregation["DEPTH"] = ["mean", "median", "max"]

    for column in PATHWAY_COLUMNS:
        if column in dataframe.columns:
            aggregation[column] = "max"

    result = dataframe.groupby("ID").agg(aggregation).reset_index()
    result.columns = [
        "_".join(column).strip("_")
        if isinstance(column, tuple)
        else column
        for column in result.columns
    ]
    return result.rename(columns={
        "_mutation_sum": "mutation_count",
        "mutation_sum": "mutation_count",
        "VAF_mean": "vaf_mean",
        "VAF_median": "vaf_median",
        "VAF_max": "vaf_max",
        "VAF_min": "vaf_min",
        "DEPTH_mean": "depth_mean",
        "DEPTH_median": "depth_median",
        "DEPTH_max": "depth_max",
    })


def make_test_data():
    clinical = pd.read_csv(CLINICAL_TEST_PATH)
    molecular = pd.read_csv(MOLECULAR_TEST_PATH)
    molecular = aggregate_molecular_data(molecular)

    test = clinical.merge(
        molecular,
        on="ID",
        how="left",
        validate="one_to_one",
    )

    molecular_features = [
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
    available_features = [
        column for column in molecular_features
        if column in test.columns
    ]
    test[available_features] = test[available_features].fillna(0)
    return test


def make_survival_target(dataframe):
    return np.array(
        [
            (bool(event), float(time))
            for event, time in zip(
                dataframe["OS_STATUS"],
                dataframe["OS_YEARS"],
            )
        ],
        dtype=[("event", "?"), ("time", "<f8")],
    )


def main():
    train = pd.read_csv(TRAIN_PATH)
    train = train.dropna(subset=["OS_YEARS", "OS_STATUS"]).copy()
    test = make_test_data()

    ignored_columns = ["ID", "OS_YEARS", "OS_STATUS"]
    feature_columns = [
        column for column in train.columns
        if column not in ignored_columns
    ]

    categorical_columns = train[feature_columns].select_dtypes(
        include=["object", "string"]
    ).columns.tolist()
    numeric_columns = [
        column for column in feature_columns
        if column not in categorical_columns
    ]

    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_columns,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "one_hot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]),
            categorical_columns,
        ),
    ])

    X_train = preprocessor.fit_transform(train[feature_columns])
    X_test = preprocessor.transform(test[feature_columns])

    model = CoxnetSurvivalAnalysis(
        l1_ratio=0.5,
        alphas=[0.1],
        max_iter=10000,
    )
    model.fit(X_train, make_survival_target(train))

    submission = pd.DataFrame({
        "ID": test["ID"],
        "risk_score": model.predict(X_test),
    })
    submission.to_csv(OUTPUT_PATH, index=False)

    print(f"training patients: {len(train)}")
    print(f"submission patients: {len(submission)}")
    print(submission.head())
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
