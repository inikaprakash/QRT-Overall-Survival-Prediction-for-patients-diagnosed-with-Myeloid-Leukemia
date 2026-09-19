import pandas as pd
import numpy as np
import re


def parse_cytogenetics(value):
    # turn one cytogenetic result into useful model features


    if pd.isna(value):
        value = ""

    value = str(value).strip()

    missing = value == ""

    unknown = (
        missing
        or value.lower() in {
            "none",
            "nan",
            "unknown",
            "na",
        }
    )

    s = re.sub(r"\s+", "", value.lower())

    features = {
        "cyto_unknown": int(unknown),
        "cyto_normal": 0,
        "cyto_abnormal": 0,

        "is_mosaic": 0,
        "n_clones": 1,
        "n_abnormal_clones": 0,
        "abnormal_cell_pct": np.nan,

        "n_cyto_events": 0,
        "is_complex": 0,
        "is_monosomal": 0,
        "n_autosomal_losses": 0,

        "has_deletion": 0,
        "has_translocation": 0,
        "has_inversion": 0,
        "has_duplication": 0,
        "has_additional_material": 0,
        "has_marker": 0,
        "has_gain": 0,
        "has_loss": 0,

        "has_minus5": 0,
        "has_5q_del": 0,

        "has_minus7": 0,

        "has_17p_abn": 0,

        "has_20q_del": 0,

        "has_chr3_abn": 0,
        "has_inv3": 0,
        "has_t_3_3": 0,
        "has_3q21": 0,
        "has_3q26": 0,
    }

    # stop here when there is no usable cytogenetic result
    if unknown:
        return features

    # split mosaic results into separate clones
    clones = s.split("/")

    features["n_clones"] = len(clones)
    features["is_mosaic"] = int(len(clones) > 1)

    total_cells = 0
    abnormal_cells = 0

    for clone in clones:
        count_match = re.search(r"\[(\d+)\]", clone)

        if count_match:
            cell_count = int(count_match.group(1))
            total_cells += cell_count
        else:
            cell_count = None

        clone_clean = re.sub(r"\[\d+\]", "", clone)
        has_abnormality = bool(
            re.search(
                r"""
                del\(
                |dup\(
                |inv\(
                |t\(
                |dic\(
                |der\(
                |add\(
                |i\(
                |idic\(
                |r\(
                |hsr\(
                |ins\(
                |-[1-9][0-9]*
                |\+[1-9][0-9]*
                |mar
                |dmin
                """,
                clone_clean,
                flags=re.VERBOSE,
            )
        )

        if has_abnormality:
            features["n_abnormal_clones"] += 1

            if cell_count is not None:
                abnormal_cells += cell_count

    if total_cells > 0:
        features["abnormal_cell_pct"] = (
            100.0 * abnormal_cells / total_cells
        )

    # check whether the result contains any clear abnormality
    has_structural_abnormality = bool(
        re.search(
            r"""
            del\(
            |dup\(
            |inv\(
            |t\(
            |dic\(
            |der\(
            |add\(
            |i\(
            |idic\(
            |r\(
            |hsr\(
            |ins\(
            """,
            s,
            flags=re.VERBOSE,
        )
    )

    has_numeric_abnormality = bool(
        re.search(
            r"(?<![a-z])-[1-9][0-9]*(?![a-z])|"
            r"(?<![a-z])\+[1-9][0-9]*(?![a-z])",
            s,
        )
    )

    has_other_abnormality = (
        "complex/other" in s
        or "mar" in s
        or "dmin" in s
    )

    has_explicit_abnormality = (
        has_structural_abnormality
        or has_numeric_abnormality
        or has_other_abnormality
    )

    # label the karyotype as normal or abnormal
    if not has_explicit_abnormality:
        if re.match(r"^46,[xy_]+$", s):
            features["cyto_normal"] = 1

        elif re.match(r"^46[-~]46,[xy_]+$", s):
            features["cyto_normal"] = 1

        else:
            features["cyto_abnormal"] = 1

    else:
        features["cyto_abnormal"] = 1

    # add general abnormality flags
    features["has_deletion"] = int("del(" in s)

    features["has_translocation"] = int(
        bool(re.search(r"t\(", s))
    )

    features["has_inversion"] = int(
        "inv(" in s
    )

    features["has_duplication"] = int(
        "dup(" in s
    )

    features["has_additional_material"] = int(
        "add(" in s
    )

    features["has_marker"] = int(
        bool(
            re.search(
                r"(?:\+mar|-?mar|mar\d*)",
                s
            )
        )
    )

    features["has_gain"] = int(
        bool(
            re.search(
                r"(?<![a-z])\+[1-9][0-9]*(?![a-z])",
                s
            )
        )
    )

    features["has_loss"] = int(
        bool(
            re.search(
                r"(?<![a-z])-[1-9][0-9]*(?![a-z])",
                s
            )
        )
    )

    def chromosome_losses(chromosome):
        return bool(
            re.search(
                rf"(?<![\w])-({chromosome})(?![\w])",
                s
            )
        )

    # add chromosome loss and disease-related flags
    features["has_minus5"] = int(
        chromosome_losses(5)
    )

    features["has_minus7"] = int(
        chromosome_losses(7)
    )

    autosomal_losses = re.findall(
        r"(?<![\w])-(\d+)(?![\w])",
        s
    )

    autosomal_losses = [
        int(x)
        for x in autosomal_losses
        if 1 <= int(x) <= 22
    ]

    unique_autosomal_losses = sorted(
        set(autosomal_losses)
    )

    features["n_autosomal_losses"] = len(
        unique_autosomal_losses
    )

    features["is_monosomal"] = int(
        features["n_autosomal_losses"] >= 2
    )

    features["has_5q_del"] = int(
        bool(
            re.search(
                r"del\(5\)\([^)]*q",
                s
            )
        )
    )

    features["has_17p_abn"] = int(
        bool(
            re.search(
                r"""
                (?:del|add|dup|inv)\(17\)\([^)]*p
                |
                i\(17\)
                |
                idic\(17\)
                """,
                s,
                flags=re.VERBOSE,
            )
        )
    )

    features["has_20q_del"] = int(
        bool(
            re.search(
                r"del\(20\)\([^)]*q",
                s
            )
        )
    )

    if "del(20q" in s:
        features["has_20q_del"] = 1

    features["has_chr3_abn"] = int(
        bool(
            re.search(
                r"""
                (?:del|add|dup|inv)\(3\)
                |
                t\(3[;:]
                |
                t\([^;]+;3[;)]
                |
                der\(3
                |
                dic\(3
                |
                i\(3\)
                """,
                s,
                flags=re.VERBOSE,
            )
        )
    )

    features["has_inv3"] = int(
        bool(
            re.search(
                r"inv\(3\)",
                s
            )
        )
    )

    features["has_t_3_3"] = int(
        bool(
            re.search(
                r"t\(3[;:]3",
                s
            )
        )
    )

    features["has_3q21"] = int(
        bool(
            re.search(
                r"""
                3\)\([^)]*q21
                |
                3[;:][^)]*3\)\(q21
                |
                q21
                """,
                s,
                flags=re.VERBOSE,
            )
        )
    )

    features["has_3q26"] = int(
        bool(
            re.search(
                r"""
                3\)\([^)]*q26
                |
                q26
                """,
                s,
                flags=re.VERBOSE,
            )
        )
    )

    # count events so complex karyotypes can be identified
    event_patterns = [
        r"del\(",
        r"dup\(",
        r"inv\(",
        r"t\(",
        r"dic\(",
        r"der\(",
        r"add\(",
        r"i\(",
        r"idic\(",
        r"r\(",
        r"hsr\(",
        r"ins\(",

        r"(?<![\w])-[1-9][0-9]*(?![\w])",
        r"(?<![\w])\+[1-9][0-9]*(?![\w])",
        r"(?:\+mar\d*|(?<![a-z])mar\d*)",
        r"dmin",
    ]

    event_count = 0

    for pattern in event_patterns:
        event_count += len(
            re.findall(pattern, s)
        )

    features["n_cyto_events"] = event_count

    features["is_complex"] = int(
        event_count >= 3
        or "complex/other" in s
    )

    return features


def add_cytogenetic_features(
    df,
    cytogenetic_col="CYTOGENETICS"
):

    # keep the original dataframe unchanged
    df = df.copy()

    if cytogenetic_col not in df.columns:
        print(
            f"Warning: '{cytogenetic_col}' not found. "
            "No cytogenetic features added."
        )
        return df

    # parse every row and turn the results into columns
    parsed = df[cytogenetic_col].apply(
        parse_cytogenetics
    )

    cytogenetic_features = pd.DataFrame(
        parsed.tolist(),
        index=df.index
    )

    existing_generated = [
        col
        for col in cytogenetic_features.columns
        if col in df.columns
    ]

    if existing_generated:
        df = df.drop(
            columns=existing_generated
        )

    df = pd.concat(
        [
            df,
            cytogenetic_features
        ],
        axis=1
    )

    return df


def clean_clinical(df):
    # clean missing values and convert numeric columns
    df = df.copy()
    missing_values = [
        "",
        " ",
        "NA",
        "N/A",
        "na",
        "n/a",
        "None",
        "none",
        "NULL",
        "null",
        "?",
    ]

    df = df.replace(
        missing_values,
        np.nan
    )

    for col in df.columns:
        if col in {
            "ID",
            "CYTOGENETICS",
        }:
            continue

        if df[col].dtype == "object":

            converted = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            non_null = df[col].notna().sum()

            if non_null > 0:

                conversion_rate = (
                    converted.notna().sum()
                    / non_null
                )

                if conversion_rate >= 0.95:
                    df[col] = converted

    return df


def clinical_features(train, test):
    # run the same feature engineering on train and test
    train = train.copy()
    test = test.copy()

    # clean both datasets first
    train = clean_clinical(train)
    test = clean_clinical(test)

    # create the cytogenetic features
    train = add_cytogenetic_features(
        train,
        cytogenetic_col="CYTOGENETICS"
    )

    test = add_cytogenetic_features(
        test,
        cytogenetic_col="CYTOGENETICS"
    )

    # make sure test has every feature found in train
    cytogenetic_cols = [
        col
        for col in train.columns
        if (
            col.startswith("cyto_")
            or col.startswith("has_")
            or col.startswith("is_")
            or col.startswith("n_")
            or col.startswith("abnormal_")
        )
    ]

    for col in cytogenetic_cols:

        if col not in test.columns:
            test[col] = 0

    print("=" * 70)
    print("CYTOGENETIC FEATURE SUMMARY")
    print("=" * 70)

    print(
        f"Number of cytogenetic features: "
        f"{len(cytogenetic_cols)}"
    )

    print("\nFeatures:")
    for col in cytogenetic_cols:
        print(f"  {col}")

    print("\nTrain feature summary:")

    summary_cols = [
        col
        for col in cytogenetic_cols
        if col in train.columns
    ]

    print(
        train[summary_cols]
        .describe()
        .T
        .round(2)
    )

    print("\nCytogenetic status:")

    status_counts = pd.DataFrame({
        "Unknown": [
            train["cyto_unknown"].sum()
        ],
        "Normal": [
            train["cyto_normal"].sum()
        ],
        "Abnormal": [
            train["cyto_abnormal"].sum()
        ],
    })

    print(status_counts.to_string(index=False))

    print("\nMosaic cases:")
    print(
        int(train["is_mosaic"].sum())
    )

    print("\nComplex karyotype cases:")
    print(
        int(train["is_complex"].sum())
    )

    print("\nMonosomal cases:")
    print(
        int(train["is_monosomal"].sum())
    )

    print("\nMean autosomal chromosome losses:")
    print(
        round(
            train["n_autosomal_losses"].mean(),
            3
        )
    )

    if "CYTOGENETICS" in train.columns:

        print("\nRaw CYTOGENETICS examples:")
        print(
            train["CYTOGENETICS"]
            .dropna()
            .head(10)
            .to_string(index=False)
        )

    print("=" * 70)

    return train, test


train = pd.read_csv(
    "./X_train/clinical_train.csv"
)

test = pd.read_csv(
    "./X_test/clinical_test.csv"
)

clinical_train, clinical_test = clinical_features(
    train,
    test
)

print("\nFinal train shape:", clinical_train.shape)
print("Final test shape:", clinical_test.shape)

cyto_cols = [
    col
    for col in clinical_train.columns
    if (
        col.startswith("cyto_")
        or col.startswith("has_")
        or col.startswith("is_")
        or col.startswith("n_")
        or col.startswith("abnormal_")
    )
]

print(cyto_cols)

clinical_train.to_csv(
    "./X_train/clinical_train_processed.csv",
    index=False
)

clinical_test.to_csv(
    "./X_test/clinical_test_processed.csv",
    index=False
)

print("processed clinical files saved")