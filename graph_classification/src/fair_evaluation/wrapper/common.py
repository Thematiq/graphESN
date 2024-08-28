from abc import abstractmethod, ABC


class ScikitFriendlyModelWrapper(ABC):
    def __init__(self, **kwargs):
        self._params = kwargs
        if len(kwargs) > 0:
            self._init_model()

    @abstractmethod
    def fit(self, data, y, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, data, *args, **kwargs):
        raise NotImplementedError()

    def get_params(self, deep=True):
        return self._params

    def set_params(self, **params):
        for key in self._params:
            if key not in params:
                params[key] = self._params[key]
        self._params = params
        self._init_model()
        return self

    @abstractmethod
    def _init_model(self):
        raise NotImplementedError()
