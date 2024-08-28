import pandas as pd
import xgboost as xgb
import torch
import numpy as np
import sys

from reservoirpy.nodes import Input, Reservoir
from graph_esn.reservoir.graph import GroupOfGESNCell, DeepGESNCell
from auto_esn.esn.reservoir.initialization import WeightInitializer
from auto_esn.esn.reservoir.activation import tanh
from tqdm.auto import trange, tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import RandomizedSearchCV
from joblib import Parallel, delayed

EMBEDDING_SIZE=16

NO_GROUPS = 1
NO_LAYERS = 5
activation = tanh()
initializer = WeightInitializer()

def top_k_cosine_similarity_matrix(similarity_matrix, k=5):
    """
    Builds a boolean matrix where for each row, the top k values in the cosine similarity matrix 
    are set to True, and the rest are set to False.

    Args:
    similarity_matrix (np.ndarray): Cosine similarity matrix.
    k (int): Number of top values to set as True for each row.

    Returns:
    np.ndarray: Boolean matrix with True for top k values per row and False otherwise.
    """
    top_k_matrix = np.zeros_like(similarity_matrix, dtype=bool)

    for i in range(similarity_matrix.shape[0]):
        top_k_indices = np.argpartition(-similarity_matrix[i], k)[:k]

        top_k_matrix[i, top_k_indices] = True

    return top_k_matrix


def normalized_symmetric_laplacian(A):
    """
    Computes the normalized symmetric Laplacian matrix.

    Args:
        A (np.ndarray): Adjacency matrix.

    Returns:
        np.ndarray: Normalized symmetric Laplacian matrix.
    """
    A = np.array(A)
    
    D = np.diag(np.sum(A, axis=1))
    
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    
    L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    
    return L


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, tqdm_kwargs={}, *args, **kwargs):
        self.tqdm_kwargs = tqdm_kwargs
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total, **self.tqdm_kwargs) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


class TempGESN:
        def __init__(self, no_features, window_size, gesn_embedding, esn_embeddings, top_k, xgboost_params, gesn_no_layers=6, esn_no_layers=3, tqdm_kwargs={}, 
                     n_jobs=-1, use_parallel_embedding=True):
                self.parallel_embedding=use_parallel_embedding
                self.tqdm_kwargs = tqdm_kwargs
                self.n_jobs = n_jobs
                self.window_size = window_size
                self.features = no_features
                self.xgboost_params = xgboost_params
                self.top_k = top_k
                self.gesn_reservoir = GroupOfGESNCell(
                input_size=window_size, hidden_size=gesn_embedding, groups=[
                        DeepGESNCell(input_size=window_size,
                                        hidden_size=gesn_embedding,
                                        initializer=initializer,
                                        num_layers=gesn_no_layers,
                                        activation=activation,
                                        conv_th=1e-5,
                                        max_iter=50)
                        for _ in range(NO_GROUPS)],
                activation=activation, initializer=initializer,
                conv_th=1e-5, max_iter=50)

                reservoirs = [Reservoir(esn_embeddings) for _ in range(esn_no_layers)]
                self.esn_reservoir = reservoirs[0]
                for i in range(1, len(reservoirs)):
                        self.esn_reservoir = self.esn_reservoir >> reservoirs[i]
                self.esn_reservoir = self.esn_reservoir & Input() >> reservoirs # Deep ESN

                self.head = None

        @staticmethod
        def __single_embedd_step(X, esn_model, gesn_model, top_k):
            # 1. Calculate ESN embeddings
            esn_embeddings = []
            for feature in range(X.shape[0]):
                    esn_model.reset()
                    esn_embeddings.append(esn_model(X[feature, :]).flatten())
            esn_embeddings = np.stack(esn_embeddings, axis=0)

            # 2. Build local graph
            similarity = cosine_similarity(esn_embeddings)
            A = top_k_cosine_similarity_matrix(similarity, top_k)
            L = torch.from_numpy(normalized_symmetric_laplacian(A)).float()

            X = torch.from_numpy(X).float()

            # 3. Calculate GESN embeddings

            gesn_model.reset_hidden()

            # print(X.shape)
            # print(L.shape) # <- should be 320x320

            gesn_embedding = gesn_model(X, L).numpy()

            # GESN embedding is [no_feat, embedd]
            # ESN embedding is [no_feat, embedd]

            # 4. Concat

            embedding = np.concatenate([esn_embeddings, gesn_embedding], axis=1)
            return embedding

        def embed_data(self, X_batch):
                # Digest batch

                if self.parallel_embedding:
                    return np.stack(
                        ProgressParallel(n_jobs=self.n_jobs, total=X_batch.shape[0], tqdm_kwargs=self.tqdm_kwargs)(
                                delayed(self.__single_embedd_step)(X_batch[i, ...], self.esn_reservoir, self.gesn_reservoir, self.top_k)
                                for i in range(X_batch.shape[0])
                        ),
                        axis=0
                    )
                
                embdeds = []

                for i in trange(X_batch.shape[0]):
                     embdeds.append(self.__single_embedd_step(X_batch[i, ...], self.esn_reservoir, self.gesn_reservoir, self.top_k))

                return np.stack(embdeds, axis=0)

        def forward(self, X_batch, y_batch):
                # X_batch size is [no_samples, features, window_size]
                # y_batch ssize is [no_samples, features]
                
                embeddings = self.embed_data(X_batch)

                # Unroll for XGBoost

                X_input = embeddings.reshape(X_batch.shape[0] * X_batch.shape[1], -1)
                y_input = y_batch.reshape(X_batch.shape[0] * X_batch.shape[1], -1)

                # 5. Increment learning XGBoost

                params = {
                        'min_child_weight': [1, 5, 10],
                        'gamma': [0.5, 1, 1.5, 2, 5],
                        'subsample': [0.6, 0.8, 1.0],
                        'colsample_bytree': [0.6, 0.8, 1.0],
                        'max_depth': [3, 4, 5]
                        }
                
                reg = xgb.XGBRegressor(learning_rate=1e-3, n_estimators=600, n_jobs=-1)
                self.head = RandomizedSearchCV(reg, params, scoring='neg_mean_absolute_error', n_jobs=-1, n_iter=5).fit(X_input, y_input)

        def predict(self, X_batch):
                # X_batch size is [no_samples, features, window_size]
                
                embeddings = self.embed_data(X_batch)

                # Unrill for XGBoost
                X_input = embeddings.reshape(X_batch.shape[0] * X_batch.shape[1], -1)

                preds = self.head.predict(X_input)

                return preds.reshape(X_batch.shape[0], X_batch.shape[1])
        
        def predict_horizon(self, X_batch, horizon=3):
            preds = []
            for _ in range(horizon):
                y_pred = self.predict(X_batch)
                X_batch = X_batch[:, :, 1:]
                X_batch = np.concatenate([X_batch, y_pred.reshape(X_batch.shape[0], X_batch.shape[1], 1)], axis=2)
                preds.append(y_pred)
            return np.stack(preds, axis=2)



def root_relative_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate the Root Relative Squared Error (RSE) between the predicted and true values
    for a batch of samples using NumPy arrays. The data is assumed to have the shape 
    [no samples, no features, horizon].

    Args:
    y_pred (np.ndarray): Predicted values, shape [no samples, no features, horizon]
    y_true (np.ndarray): True values, shape [no samples, no features, horizon]

    Returns:
    float: The computed RSE, averaged across all samples.
    """
    numerator = np.sum((y_pred - y_true) ** 2, axis=(1, 2))
    
    denominator = np.sum((y_true - np.mean(y_true, axis=2, keepdims=True)) ** 2, axis=(1, 2))
    
    rse = np.sqrt(numerator / (denominator + 1e-8))
    
    return np.mean(rse)


def empirical_correlation_coefficient(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate the Empirical Correlation Coefficient (CORR) between the predicted and true values
    for a batch of samples using NumPy arrays. The data is assumed to have the shape 
    [no_samples, no_features, horizon].

    Args:
    y_pred (np.ndarray): Predicted values, shape [no_samples, no_features, horizon]
    y_true (np.ndarray): True values, shape [no_samples, no_features, horizon]

    Returns:
    float: The computed CORR, averaged across all samples and features.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    y_true_mean = np.mean(y_true, axis=2, keepdims=True)
    y_pred_mean = np.mean(y_pred, axis=2, keepdims=True)

    numerator = np.sum((y_true - y_true_mean) * (y_pred - y_pred_mean), axis=2)

    denominator_true = np.sqrt(np.sum((y_true - y_true_mean)**2, axis=2))
    denominator_pred = np.sqrt(np.sum((y_pred - y_pred_mean)**2, axis=2))
    denominator = denominator_true * denominator_pred

    corr = numerator / (denominator + 1e-8)

    return np.mean(corr)


DATASET = sys.argv[1]
print(f"Dataset {DATASET}")

N_JOBS = -1
WINDOW_SIZE = 256

def build_batch(X, window_size):
    X_dataset = []
    y_dataset = []

    for i in range(0, X.shape[0] - 1 - window_size):
        X_dataset.append(X.iloc[i : i + window_size].values.T)
        y_dataset.append(X.iloc[i + window_size + 1].values)

    X_dataset = np.stack(X_dataset, axis=0)
    y_dataset = np.stack(y_dataset, axis=0)
    return X_dataset, y_dataset

df = pd.read_csv(f"multivariate-time-series-data/{DATASET}/{DATASET}.txt.gz", header=None, compression="gzip")

splitting_point = int(len(df.index) * 0.9)

X_train = df.iloc[:splitting_point]
X_test = df.iloc[splitting_point:]

del df

model = TempGESN(
        no_features=X_train.shape[1],
        window_size=WINDOW_SIZE,
        gesn_embedding=32,
        esn_embeddings=128,
        esn_no_layers=6,
        gesn_no_layers=6,
        top_k=max(int(X_train.shape[1] * 0.1), 2),
        tqdm_kwargs={'leave': False},
        n_jobs=-1,
        xgboost_params={
            'objective':'reg:squarederror'
        }
)

def build_val_batch(X, horizon, window_size):
    xs = []
    ys = []

    for i in range(window_size, X.shape[0] - horizon, 1):
        X_part = X.iloc[i - window_size : i].T
        y_part = X.iloc[i: i + horizon].T
        xs.append(X_part)
        ys.append(y_part)

    X_res = np.stack(xs, axis=0)
    y_res = np.stack(ys, axis=0)

    return X_res, y_res


X_train, y_train = build_batch(X_train, WINDOW_SIZE)

model.forward(X_train, y_train)

del X_train, y_train

import pickle

with open(f"{DATASET}_model.pickle", 'wb') as f:
    pickle.dump(model, f)


for horizon in [3, 6, 12, 24]:
    X_input, y_test = build_val_batch(X_test, horizon, WINDOW_SIZE)
    y_pred = model.predict_horizon(X_input, horizon)

    corr = empirical_correlation_coefficient(y_pred, y_test)
    rse = root_relative_squared_error(y_pred, y_test)

    print(f"Correlation: {corr:.3f}, RSE: {rse:.3f}")

    with open(f'{DATASET}_res.txt', 'a') as f:
        f.write(f"{horizon} {corr:.4f} {rse:.4f}\n")


