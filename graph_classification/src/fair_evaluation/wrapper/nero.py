from .common import ScikitFriendlyModelWrapper

from nero.embedding.pipelines import create_bin_generators, create_compressor, create_normaliser
from nero.embedding.embedders import NeroEmbedder
from nero.converters.tudataset import TUDatasetDescription
from nero.pipeline.transform import resize

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class NEROWrapper(ScikitFriendlyModelWrapper):
    def __init__(self, **kwargs):
        self._model = None
        super().__init__(**kwargs)

    def fit(self, data, y, *args, **kwargs):
        desc = kwargs['fit_params']['desc']
        self._model = self.__build_model(desc)
        self._model.fit(data, y)
        return self

    def predict(self, data, *args, **kwargs):
        return self._model.predict(data)

    def _init_model(self):
        pass

    def __build_model(self, desc):
        digitiser_tag, compressor_tag, normaliser_tag = [*self._params['tag']][:3]

        embedder = NeroEmbedder(
            bin_generators=create_bin_generators(desc, digitiser_tag, self._params["digitiser_parameter"]),
            relation_order_compressor=create_compressor(compressor_tag, self._params["compressor_parameter"]),
            jobs_no=1,
            disable_tqdm=True,
        )
        classifier = ExtraTreesClassifier(
            n_estimators=5000,
            n_jobs=1,
            verbose=False
        )
        return Pipeline([
            ('embed', embedder),
            ('resize_to_smallest_common_shape', resize.ToSmallestCommonShape(disable_tqdm=True)),
            ('normalise', create_normaliser(normaliser_tag)),
            ('flatten', resize.FlattenUpperTriangles(disable_tqdm=True)),
            ('scale', StandardScaler()),
            ('classify', classifier),
        ])
