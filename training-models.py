import random

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.ensemble import (
	GradientBoostingSurvivalAnalysis,
	RandomSurvivalForest,
)
from sksurv.linear_model import (
	CoxPHSurvivalAnalysis,
	CoxnetSurvivalAnalysis,
)
from sksurv.metrics import concordance_index_censored


RANDOM_STATE = 42
TRAIN_PATH = "./X_train/train_60_processed.csv"
VALIDATION_PATH = "./X_train/validation_20_processed.csv"
TEST_PATH = "./X_train/test_20_processed.csv"
RESULTS_PATH = "./X_train/model_results.csv"


def make_survival_target(df):
	return np.array(
		[
			(bool(event), float(time))
			for event, time in zip(df["OS_STATUS"], df["OS_YEARS"])
		],
		dtype=[("event", "?"), ("time", "<f8")],
	)


def load_data():
	train = pd.read_csv(TRAIN_PATH)
	validation = pd.read_csv(VALIDATION_PATH)
	test = pd.read_csv(TEST_PATH)

	target_columns = ["OS_YEARS", "OS_STATUS"]
	for dataframe in (train, validation, test):
		dataframe.dropna(subset=target_columns, inplace=True)

	return train, validation, test


def make_preprocessor(train):
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

	numeric_pipeline = Pipeline([
		("imputer", SimpleImputer(strategy="median")),
		("scaler", StandardScaler()),
	])
	categorical_pipeline = Pipeline([
		("imputer", SimpleImputer(strategy="most_frequent")),
		(
			"one_hot",
			OneHotEncoder(
				handle_unknown="ignore",
				sparse_output=False,
			),
		),
	])

	preprocessor = ColumnTransformer([
		("numeric", numeric_pipeline, numeric_columns),
		("categorical", categorical_pipeline, categorical_columns),
	])

	return preprocessor, feature_columns


def transform_data(preprocessor, feature_columns, dataframe):
	return preprocessor.transform(dataframe[feature_columns]).astype(np.float64)


def c_index(target, risk_scores):
	return concordance_index_censored(
		target["event"],
		target["time"],
		np.asarray(risk_scores).reshape(-1),
	)[0]


def fit_sklearn_models(X_train, y_train):
	models = {
		"cox_ph": CoxPHSurvivalAnalysis(alpha=0.1),
		"elastic_net_cox": CoxnetSurvivalAnalysis(
			l1_ratio=0.5,
			alphas=[0.1],
			max_iter=10000,
		),
		"random_survival_forest": RandomSurvivalForest(
			n_estimators=300,
			min_samples_split=10,
			min_samples_leaf=5,
			max_features="sqrt",
			n_jobs=-1,
			random_state=RANDOM_STATE,
		),
		"gradient_boosted_survival": GradientBoostingSurvivalAnalysis(
			loss="coxph",
			n_estimators=200,
			learning_rate=0.03,
			max_depth=2,
			min_samples_leaf=10,
			random_state=RANDOM_STATE,
		),
	}

	fitted_models = {}
	for name, model in models.items():
		print(f"training {name}...")
		model.fit(X_train, y_train)
		fitted_models[name] = model

	return fitted_models


class CoxNeuralNetwork(torch.nn.Module):
	def __init__(self, input_size):
		super().__init__()
		self.network = torch.nn.Sequential(
			torch.nn.Linear(input_size, 64),
			torch.nn.ReLU(),
			torch.nn.Dropout(0.15),
			torch.nn.Linear(64, 32),
			torch.nn.ReLU(),
			torch.nn.Linear(32, 1),
		)

	def forward(self, features):
		return self.network(features).reshape(-1)


def cox_loss(risk_scores, target):
	order = torch.argsort(target[:, 1], descending=True)
	ordered_risk = risk_scores[order]
	ordered_events = target[order, 0]
	log_risk_sets = torch.logcumsumexp(ordered_risk, dim=0)
	return -((ordered_risk - log_risk_sets) * ordered_events).sum() / (
		ordered_events.sum() + 1e-8
	)


def fit_neural_network(X_train, y_train, epochs=300):
	torch.manual_seed(RANDOM_STATE)
	model = CoxNeuralNetwork(X_train.shape[1])
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

	features = torch.tensor(X_train, dtype=torch.float32)
	target = torch.tensor(
		np.column_stack([
			y_train["event"].astype(np.float32),
			y_train["time"].astype(np.float32),
		])
	)

	model.train()
	for _ in range(epochs):
		optimizer.zero_grad()
		loss = cox_loss(model(features), target)
		loss.backward()
		optimizer.step()

	model.eval()
	return model


def neural_predictions(model, features):
	with torch.no_grad():
		return model(torch.tensor(features, dtype=torch.float32)).numpy()


def main():
	random.seed(RANDOM_STATE)
	np.random.seed(RANDOM_STATE)

	train, validation, test = load_data()
	y_train = make_survival_target(train)
	y_validation = make_survival_target(validation)
	y_test = make_survival_target(test)

	preprocessor, feature_columns = make_preprocessor(train)
	X_train = preprocessor.fit_transform(train[feature_columns])
	X_validation = transform_data(preprocessor, feature_columns, validation)
	X_test = transform_data(preprocessor, feature_columns, test)

	print(f"train shape: {X_train.shape}")
	print(f"validation shape: {X_validation.shape}")
	print(f"test shape: {X_test.shape}")

	models = fit_sklearn_models(X_train, y_train)
	neural_network = fit_neural_network(X_train, y_train)
	models["survival_neural_network"] = neural_network

	results = []
	for name, model in models.items():
		if name == "survival_neural_network":
			validation_scores = neural_predictions(model, X_validation)
			test_scores = neural_predictions(model, X_test)
		else:
			validation_scores = model.predict(X_validation)
			test_scores = model.predict(X_test)

		results.append({
			"model": name,
			"validation_c_index": c_index(y_validation, validation_scores),
			"test_c_index": c_index(y_test, test_scores),
		})

	results = pd.DataFrame(results).sort_values(
		"validation_c_index",
		ascending=False,
	)
	results.to_csv(RESULTS_PATH, index=False)

	print("\nmodel results:")
	print(results.to_string(index=False))
	print(f"\nsaved: {RESULTS_PATH}")


if __name__ == "__main__":
	main()
