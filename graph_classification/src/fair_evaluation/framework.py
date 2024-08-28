import os

import pandas as pd
import torch.multiprocessing as mp

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import get_scorer
from tqdm import tqdm
from time import time
from datetime import datetime

from fair_evaluation.eval.dataloader import default_loader
from fair_evaluation.const import RANDOM_STATE


def perform_inner_evaluation(
        model,
        hyperparam_grid,
        data, y,
        performance: str = 'accuracy',
        inner_splits: int = 10,
        n_jobs: int = -1,
        fit_args=None,
        allow_for_exception=False,
):
    if fit_args is None:
        fit_args = dict()

    splitter = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_res = GridSearchCV(
        model,
        hyperparam_grid,
        cv=splitter,
        scoring=performance,
        verbose=10,
        error_score=-1 if allow_for_exception else "raise",
        n_jobs=n_jobs,
    ).fit(data, y, fit_params=fit_args)

    params = cv_res.best_params_
    score = cv_res.best_score_

    training_start = time()
    best_model = type(model)(**params).fit(data, y, fit_params=fit_args)
    training_end = time()
    training_time = training_end - training_start

    return best_model, training_time, score, params


def measure_time(func):
    start = time()
    res = func()
    end = time()
    return (end - start), res


def perform_fair_evaluation(
        model,
        hyperparam_grid,
        dataset: str,
        root_dir: str,
        performance='accuracy',
        inner_splits: int = 10,
        outer_splits: int = 10,
        n_jobs:int = -1,
        loader=default_loader,
        partial_output_loc: str = None,
        partial_folds: list = None,
        fold_dir: str = None,
        allow_for_exception=False,
):
    mp.set_start_method('spawn', force=True)
    load_time, (data, y, fit_args) = measure_time(lambda: loader(dataset, root_dir))

    print(f"Loading data took {load_time:.3} seconds")

    results = pd.DataFrame(columns=["fold", "val_score", "test_score", "time", "params"])

    splitter = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    loop = tqdm(splitter.split(data, y), total=outer_splits)

    for i, (train_index, test_index) in enumerate(loop):
        if partial_folds is not None and i not in partial_folds:
            print(f"Skipping {i} outer fold accordingly to the partial folds")
            continue

        if isinstance(data, list):
            data_train = [data[i] for i in train_index]
            data_test = [data[i] for i in test_index]
            y_train = [y[i] for i in train_index]
            y_test = [y[i] for i in test_index]
        else:
            data_train, data_test = data[train_index], data[test_index]
            y_train, y_test = y[train_index], y[test_index]

        best_model, train_time, val_score, params = perform_inner_evaluation(model, hyperparam_grid, data_train,
                                                                             y_train, performance, inner_splits, 
                                                                             n_jobs, fit_args, allow_for_exception)

        try:
            scoring_f = get_scorer(performance)
            y_pred = best_model.predict(data_test)

            score = scoring_f._score_func(y_true=y_test, y_pred=y_pred)
        except Exception as e:
            if allow_for_exception:
                score = -1
                print(f"Exception in testing!! Cause - {e}")
            else:
                raise e
        
        scores.append(score)
        loop.set_description(f"\n Report time {datetime.now()} Last iteration score: {score}," 
                             f"inner score: {val_score}, training time: {train_time}")

        results.loc[len(results.index)] = {
            'fold': i,
            'val_score': val_score,
            'test_score': score,
            'time': train_time,
            'params': params
        }

        if partial_output_loc is not None:
            results.to_csv(partial_output_loc)

        if fold_dir is not None:
            os.makedirs(fold_dir, exist_ok=True)
            results.iloc[len(results.index) - 1].to_csv(f"{fold_dir}/{i}.fold")

    return results
