from .common import ScikitFriendlyModelWrapper

from LTP.feature_extraction import extract_features, calculate_features_matrix
from LTP.models import get_model


class LTPWrapper(ScikitFriendlyModelWrapper):
    LDP_PARAMS_NAME = ["n_bins",  "normalization", "aggregation", "log_degree"]
    FEATURE_EXTRACT_PARAMS_NAME = ["degree_sum", "shortest_paths", "edge_betweenness", "jaccard_index", "local_degree_score"]

    def __init__(self, **kwargs):
        self._ldp_params = None
        self._extract_params = None
        self._model = None
        super().__init__(**kwargs)

    def fit(self, data, y, *args, **kwargs):
        features = extract_features(
            data,
            **self._extract_params
        )
        X_train = calculate_features_matrix(features, **self._ldp_params)
        self._model = get_model("RandomForest", tune_model_hyperparams=False, verbose=False)
        self._model.fit(X_train, y)
        return self

    def predict(self, data, *args, **kwargs):
        features = extract_features(
            data,
            **self._extract_params
        )
        X_test = calculate_features_matrix(features, **self._ldp_params)
        return self._model.predict(X_test)

    def _init_model(self):
        self._ldp_params = self.__filter(self._params, self.LDP_PARAMS_NAME)
        self._extract_params = self.__filter(self._params, self.FEATURE_EXTRACT_PARAMS_NAME)

    @staticmethod
    def __filter(dictionary, keys):
        return {k: v for k, v in dictionary.items() if k in keys}