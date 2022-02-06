"""Collection of models used. scikit-learn does not allow setting custom
loss functions or having non-binary targets, thus we implemented the models in PyTorch.
"""
import itertools
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import random

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True)


class ScaledLogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(26, 1, dtype=torch.double)

    def forward(self, x):
        return torch.sigmoid(self.layer1(x)) * 9 + 1


class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(26, 1, dtype=torch.double)

    def forward(self, x):
        return torch.sigmoid(self.layer1(x))


class LinearRegression:
    def __init__(
        self, train_set, train_targets,
    ):
        self.train_targets = -torch.log(9 / (train_targets - 1) - 1)
        self.train_set = train_set
        self.train_set = torch.hstack((torch.ones((self.train_set.shape[0], 1)), self.train_set))
        self.w = None

    def eval(self, test_set, test_targets):
        test_set = torch.hstack((torch.ones((test_set.shape[0], 1)), test_set))
        if self.w is None:
            self.fit()

        transformed_preds = test_set @ self.w
        preds = 9 * 1 / (1 + torch.exp(-transformed_preds)) + 1

        loss = (test_targets - preds).abs().mean().item()
        print("Test loss:", loss)
        return loss

    def fit(self):
        self.w = torch.linalg.pinv(self.train_set) @ self.train_targets


class DeepReLU(nn.Module):
    def __init__(self, num_layers):
        super().__init__()

        self.hidden_dim = 32

        self.layer_1 = nn.Linear(26, self.hidden_dim, dtype=torch.double)
        self.inner_layers = []

        if num_layers > 2:
            for i in range(num_layers - 2):
                self.inner_layers.append(
                    nn.Linear(self.hidden_dim, self.hidden_dim, dtype=torch.double)
                )
                self.inner_layers.append(nn.ReLU())

            self.inner_layers = nn.Sequential(*self.inner_layers)
        else:
            self.inner_layers = None

        self.layer_n = nn.Linear(self.hidden_dim, 1, dtype=torch.double)

    def forward(self, x):
        x = F.relu(self.layer_1(x))

        if self.inner_layers is not None:
            x = self.inner_layers(x)

        return torch.sigmoid(self.layer_n(x)) * 9 + 1


def train_model_variants(
    train_set_normalized,
    train_targets,
    val_set_normalized,
    val_targets,
    test_set_normalized,
    test_targets,
):
    result_dict = {}

    print("Logistic Regression, MAE")
    mae_results = train_logistic_regression(
        nn.L1Loss(),
        train_set_normalized,
        train_targets,
        val_set_normalized,
        val_targets,
        test_set_normalized,
        test_targets,
    )
    print("Logistic Regression, MSE")
    mse_results = train_logistic_regression(
        nn.MSELoss(),
        train_set_normalized,
        train_targets,
        val_set_normalized,
        val_targets,
        test_set_normalized,
        test_targets,
    )
    print("Logistic Regression, BCE")
    bce_results = train_logistic_regression(
        nn.BCELoss(),
        train_set_normalized,
        train_targets,
        val_set_normalized,
        val_targets,
        test_set_normalized,
        test_targets,
        is_bce=True,
    )

    print("Linear Regression")
    lin_reg = LinearRegression(train_set_normalized, train_targets)
    lin_reg_results = lin_reg.eval(test_set_normalized, test_targets)

    print("2-layer ReLU, MAE")
    relu_2_results = train_deep_model(
        2,
        0.003,
        train_set_normalized,
        train_targets,
        val_set_normalized,
        val_targets,
        test_set_normalized,
        test_targets,
    )

    print("4-layer ReLU, MAE")
    relu_4_results = train_deep_model(
        4,
        0.003,
        train_set_normalized,
        train_targets,
        val_set_normalized,
        val_targets,
        test_set_normalized,
        test_targets,
    )

    print("6-layer ReLU, MAE")
    relu_6_results = train_deep_model(
        6,
        0.003,
        train_set_normalized,
        train_targets,
        val_set_normalized,
        val_targets,
        test_set_normalized,
        test_targets,
    )

    result_dict["MAE"] = mae_results
    result_dict["MSE"] = mse_results
    result_dict["BCE"] = bce_results
    result_dict["LR"] = lin_reg_results
    result_dict["RELU2"] = relu_2_results
    result_dict["RELU4"] = relu_4_results
    result_dict["RELU6"] = relu_6_results
    result_dict["ground_truth"] = test_targets.squeeze().numpy()

    return result_dict


def train_logistic_regression(
    loss_fn,
    train_set_normalized,
    train_targets,
    val_set_normalized,
    val_targets,
    test_set_normalized,
    test_targets,
    is_bce=False,
):
    if is_bce:
        model = LogisticRegression()
        train_targets = (train_targets - 1) / 9
    else:
        model = ScaledLogisticRegression()

    val_fn = nn.L1Loss()
    opt = torch.optim.SGD(model.parameters(), lr=0.001)

    best_loss_in_mae = np.inf
    test_loss = None

    patience_counter = 0
    max_patience = 10

    best_val_loss = np.inf
    epoch = 0
    while (
        model.layer1.weight.grad is None
        or (torch.norm(model.layer1.weight.grad) ** 2 + model.layer1.bias.grad ** 2)
        / 27
        >= 1e-6
    ):
        pred = model(train_set_normalized)
        loss = loss_fn(pred.squeeze(), train_targets)

        with torch.no_grad():
            if is_bce:
                loss_in_mae = val_fn(pred.squeeze() * 9 + 1, train_targets * 9 + 1)
            else:
                loss_in_mae = val_fn(pred.squeeze(), train_targets)

            val_pred = model(val_set_normalized)

            if is_bce:
                val_loss = val_fn(val_pred.squeeze() * 9 + 1, val_targets)
            else:
                val_loss = val_fn(val_pred.squeeze(), val_targets)

            if val_loss.item() < best_val_loss:
                test_pred = model(test_set_normalized)

                if is_bce:
                    test_loss = val_fn(test_pred.squeeze() * 9 + 1, test_targets)
                else:
                    test_loss = val_fn(test_pred.squeeze(), test_targets)

                best_val_loss = val_loss.item()
                best_loss_in_mae = loss_in_mae
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print("Training loss (MAE):", best_loss_in_mae.item())
                    print("Test loss (MAE):", test_loss.item())
                    return (
                        best_loss_in_mae.item(),
                        test_loss.item(),
                        test_pred.squeeze().numpy(),
                    )

        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch += 1

    print("Training loss (MAE):", best_loss_in_mae.item())
    print("Test loss (MAE):", test_loss.item())
    print(model.layer1.weight)
    print(model.layer1.bias)
    return best_loss_in_mae.item(), test_loss.item(), test_pred.squeeze().numpy()


def train_deep_model(
    num_layers,
    learning_rate,
    train_set_normalized,
    train_targets,
    val_set_normalized,
    val_targets,
    test_set_normalized,
    test_targets,
):
    model = DeepReLU(num_layers)
    loss_fn = nn.MSELoss()
    val_fn = nn.L1Loss()
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    best_loss_in_mae = np.inf
    test_loss = None

    patience_counter = 0
    max_patience = 100

    best_val_loss = np.inf

    for _ in range(10000):  # 10000
        pred = model(train_set_normalized)
        loss = loss_fn(pred.squeeze(), train_targets)

        with torch.no_grad():
            loss_in_mae = val_fn(pred.squeeze(), train_targets)
            val_pred = model(val_set_normalized)
            val_loss = val_fn(val_pred.squeeze(), val_targets)

            if val_loss.item() < best_val_loss:

                test_pred = model(test_set_normalized)
                test_loss = val_fn(test_pred.squeeze(), test_targets)
                best_loss_in_mae = loss_in_mae
                best_val_loss = val_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print("Training loss (MAE):", best_loss_in_mae.item())
                    print("Test loss (MAE):", test_loss.item())
                    return (
                        best_loss_in_mae.item(),
                        test_loss.item(),
                        test_pred.squeeze().numpy(),
                    )

        opt.zero_grad()
        loss.backward()
        opt.step()

    print("Training loss (MAE):", best_loss_in_mae.item())
    print("Test loss (MAE):", test_loss.item())
    return best_loss_in_mae.item(), test_loss.item(), test_pred.squeeze().numpy()


def _clean_unknowns(row, column):
    if row[column] == "\\N":
        return None
    else:
        return row[column]


def _clean_reviews(row, column):
    if isinstance(row[column], str) and "K" in row[column]:
        if "." in row[column]:
            return int(row[column][:-3]) * 1000 + int(row[column][-2]) * 100
        else:
            return int(row[column][:-1]) * 1000
    else:
        return row[column]


def make_prediction_data(full_dataset):
    full_dataset["Critic reviews"] = full_dataset["Critic reviews"].fillna(0)
    full_dataset["User reviews"] = full_dataset["User reviews"].fillna(0)

    full_dataset["isAdult2"] = full_dataset.apply(
        lambda row: int("Adult" in row["genres"]), axis=1
    )

    # tconst was only required for joins
    # titleType is only films for us, we filtered them
    # we do not use the titles as predictors
    # endYear is None for all films
    # isAdult will be added back in a consistent format later on
    # We drop writers and directors. These are interesting features,
    # but having them as binary columns would be infeasible.
    full_dataset = full_dataset.drop(
        columns=[
            "tconst",
            "titleType",
            "primaryTitle",
            "originalTitle",
            "endYear",
            "isAdult",
            "isAdult2",
            "Gross US & Canada",
            "Opening weekend US & Canada",
            "writers",
            "directors",
        ]
    )
    full_dataset = full_dataset.dropna()

    genre_list = full_dataset["genres"].unique().tolist()
    for i, entry in enumerate(genre_list):
        genre_list[i] = entry.split(",")

    genre_set = set(itertools.chain(*genre_list))

    # News - History - Biography - Documentary --> Documentary
    # Film-Noir - Crime --> Crime
    # Western - Action --> Action
    genre_set.difference_update(
        ["News", "History", "Biography", "Film-Noir", "Western"]
    )
    transformation_dict = {
        "Documentary": ["News", "History", "Biography", "Documentary"],
        "Crime": ["Film-Noir", "Crime"],
        "Action": ["Western", "Action"],
    }
    for genre in genre_set:
        if genre not in transformation_dict:
            transformation_dict[genre] = [genre]
        full_dataset[f"is{genre}"] = full_dataset.apply(
            lambda row: int(
                any(g in row["genres"] for g in transformation_dict[genre])
            ),
            axis=1,
        )

    # Genres are added as binary predictors, thus the genres column is no longer used.
    full_dataset = full_dataset.drop(
        columns=["genres"]
    )  # "isMusical", "isFilm-Noir", "isNews", "isSport", "genres"])

    def unrated_to_not_rated(row):
        if row["Rating"] == "Unrated":
            return "Not Rated"
        else:
            return row["Rating"]

    full_dataset["Rating"] = full_dataset.apply(unrated_to_not_rated, axis=1)
    full_dataset["isRated"] = full_dataset.apply(
        lambda row: int(row["Rating"] != "Not Rated"), axis=1
    )
    full_dataset = full_dataset.drop(columns=["Rating"])

    full_dataset["startYear"] = full_dataset.apply(
        lambda row: _clean_unknowns(row, "startYear"), axis=1
    )
    full_dataset["runtimeMinutes"] = full_dataset.apply(
        lambda row: _clean_unknowns(row, "runtimeMinutes"), axis=1
    )
    full_dataset["User reviews"] = full_dataset.apply(
        lambda row: _clean_reviews(row, "User reviews"), axis=1
    )
    full_dataset["Critic reviews"] = full_dataset.apply(
        lambda row: _clean_reviews(row, "Critic reviews"), axis=1
    )

    for column in ["startYear", "runtimeMinutes", "User reviews", "Critic reviews"]:
        full_dataset[column] = pd.to_numeric(full_dataset[column])

    filtered = full_dataset.dropna()
    filtered.reset_index(inplace=True)
    filtered = filtered.drop(columns=["index"])

    test_indices = np.random.choice(
        len(filtered), replace=False, size=int(len(filtered) / 10)
    )
    test_set = filtered.iloc[test_indices]
    test_set, test_targets = (
        test_set.drop("averageRating", axis=1).to_numpy(),
        test_set["averageRating"].to_numpy(),
    )
    train_set = filtered.iloc[~filtered.index.isin(test_indices)]
    train_set, train_targets = (
        train_set.drop("averageRating", axis=1).to_numpy(),
        train_set["averageRating"].to_numpy(),
    )
    val_indices = np.random.choice(
        len(train_set), replace=False, size=int(len(train_set) / 10)
    )

    val_set = train_set[val_indices]
    val_targets = train_targets[val_indices]

    not_val_indices = [i for i in range(len(train_set)) if i not in val_indices]
    train_set = train_set[not_val_indices]
    train_targets = train_targets[not_val_indices]

    test_set = torch.from_numpy(test_set)
    test_set_normalized = (
        test_set - test_set.mean(dim=0, keepdims=True)
    ) / test_set.std(dim=0, keepdims=True)
    test_set_normalized = torch.nan_to_num(test_set_normalized, nan=0)
    test_targets = torch.from_numpy(test_targets)

    val_set = torch.from_numpy(val_set)
    val_set_normalized = (val_set - val_set.mean(dim=0, keepdims=True)) / val_set.std(
        dim=0, keepdims=True
    )
    val_set_normalized = torch.nan_to_num(val_set_normalized, nan=0)
    val_targets = torch.from_numpy(val_targets)

    train_set = torch.from_numpy(train_set)
    train_set_normalized = (
        train_set - train_set.mean(dim=0, keepdims=True)
    ) / train_set.std(dim=0, keepdims=True)
    train_set_normalized = torch.nan_to_num(train_set_normalized, nan=0)
    train_targets = torch.from_numpy(train_targets)

    return (
        (train_set_normalized, train_targets),
        (val_set_normalized, val_targets),
        (test_set_normalized, test_targets),
    )


if __name__ == "__main__":
    data = pd.read_csv("../dat/data_clean.csv", dtype={5: "object", 16: "object"})
    (
        (train_set_normalized, train_targets),
        (val_set_normalized, val_targets),
        (test_set_normalized, test_targets),
    ) = make_prediction_data(data)
    result_dict = train_model_variants(
        train_set_normalized,
        train_targets,
        val_set_normalized,
        val_targets,
        test_set_normalized,
        test_targets,
    )
    for key in result_dict:
        if key == "LR":
            continue
        result_dict[key] = (result_dict[key][0], result_dict[key][1])

    del result_dict["ground_truth"]

    print(result_dict)
