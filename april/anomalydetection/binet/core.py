# Copyright 2018 Timo Nolle
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

import numpy as np

from april.anomalydetection.basic import AnomalyDetector
from april.anomalydetection.utils.result import AnomalyDetectionResult
from april.enums import Base
from april.enums import Heuristic
from april.enums import Mode
from april.enums import Strategy
from april.fs import ModelFile


class NNAnomalyDetector(AnomalyDetector):
    """Abstract neural network based anomaly detector.

    Save and load methods are different for Keras based models.
    """
    config = None

    def __init__(self, model=None):
        super(NNAnomalyDetector, self).__init__(model=model)
        self.history = None

    def load(self, file_name):
        # load model file
        file_name = ModelFile(file_name)

        # load model
        from keras.models import load_model
        from keras.utils import CustomObjectScope
        from april.anomalydetection.binet.attention import Attention
        with CustomObjectScope({'Attention': Attention}):
            self._model = load_model(file_name.str_path)

    def _save(self, file_name):
        self.model.save(file_name)

    @staticmethod
    def model_fn(dataset, **kwargs):
        raise NotImplementedError()

    def fit(self, dataset, epochs=30, batch_size=100, validation_split=0.1, **kwargs):
        # Build model
        self._model, features, targets = self.model_fn(dataset, **self.config)

        # Train model
        self.history = self._model.fit(
            features,
            targets,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            **kwargs
        )

        return self.history

    def detect(self, dataset):
        raise NotImplementedError()


class BINet(NNAnomalyDetector):
    """Implements the BINet anomaly detection approach."""
    version = None
    supported_bases = [Base.LEGACY, Base.SCORES]
    supported_heuristics = [Heuristic.BEST, Heuristic.ELBOW_DOWN, Heuristic.ELBOW_UP,
                            Heuristic.LP_LEFT, Heuristic.LP_MEAN, Heuristic.LP_RIGHT,
                            Heuristic.MEAN, Heuristic.MEDIAN, Heuristic.RATIO, Heuristic.MANUAL]
    supported_strategies = [Strategy.SINGLE, Strategy.ATTRIBUTE, Strategy.POSITION, Strategy.POSITION_ATTRIBUTE]
    supported_modes = [Mode.BINARIZE, Mode.CLASSIFY]
    supports_attributes = True

    def __init__(self, model=None):
        super(BINet, self).__init__(model)

    @staticmethod
    def model_fn(dataset, **kwargs):
        return binet_model_fn(dataset, **kwargs)

    def detect(self, dataset):
        _, features, _ = self.model_fn(dataset, **self.config)
        scores, predictions, attentions = detect_fn(self.model, features, dataset, self.config)
        return AnomalyDetectionResult(scores=scores, predictions=predictions, attentions=attentions)


def binet_scores_fn(features, predictions):
    ps = [np.copy(p) for p in predictions]
    args = [p.argsort(axis=-1) for p in ps]
    [p.sort(axis=-1) for p in ps]
    sums = [1 - np.cumsum(p, axis=-1) for p in ps]
    indices = [np.argmax(a + 1 == features[:, :, i:i + 1], axis=-1).astype(int) for i, a in
               enumerate(args)]

    scores = np.zeros(features.shape, dtype=np.float64)
    for (i, j, k), f in np.ndenumerate(features):
        if f != 0 and k < len(predictions):
            scores[i, j, k] = sums[k][i, j][indices[k][i, j]]
    return scores


def detect_fn(model, features, dataset, config):
    # Get attention layers
    attention_layers = [l for l in model.layers if 'attention' in l.name]
    attention_weights = [l.output[1] for l in attention_layers]

    # Parameters
    num_predictions = len(model.outputs)
    num_attentions = len(attention_layers)
    num_attributes = dataset.num_attributes

    # Get config parameters from model architecture
    use_attributes = config.get('use_attributes', False)
    use_present_attributes = config.get('use_present_attributes', False)
    use_present_activity = config.get('use_present_activity', False)
    use_attentions = num_attentions > 0

    # Rebuild outputs
    outputs = []
    if use_attentions:
        # Add attention outputs
        outputs += attention_weights
    if len(outputs) > 0:
        # We have to recompile the model to include the new outputs
        from keras import Model
        model = Model(inputs=model.inputs, outputs=model.outputs + outputs)

    # Predict
    outputs = model.predict(features)

    # Predictions must be list
    if len(model.outputs) == 1:
        outputs = [outputs]

    # Split predictions
    predictions = outputs[:num_predictions]

    # Add perfect prediction for start symbol
    perfect = [f[0, :1] for f in dataset.onehot_features]
    for i in range(len(predictions)):
        predictions[i][:, 1:] = predictions[i][:, :-1]
        predictions[i][:, 0] = perfect[i]

    # Scores
    scores = binet_scores_fn(dataset.flat_features, predictions)

    # Init
    attentions = None

    # Split attentions
    if use_attentions:
        attentions = outputs[num_predictions:num_predictions + num_attentions]
        attentions = split_attentions(attentions, num_attributes, use_attributes, use_present_activity,
                                      use_present_attributes)

    return scores, predictions, attentions


def split_attentions(attentions, num_attributes, use_attributes, use_present_activity, use_present_attributes):
    n = num_attributes

    # Split attentions
    _attentions = []
    if not use_attributes:
        attentions += [np.zeros_like(attentions[0])] * ((n - 1) + (n - 1) * n)

    for i in range(n):
        _a = []
        for j in range(n):
            a = attentions[n * i + j]
            for __a in a:
                __a[np.triu_indices(a.shape[1], 1)] = -1
            # Add empty attentions for start symbol
            a = np.pad(a[:, :-1], ((0, 0), (1, 0), (0, 0)), mode='constant')
            if i != 0:
                if (use_present_attributes and i != j) or (use_present_activity and j == 0):
                    a = np.pad(a[:, :, :-1], ((0, 0), (0, 0), (1, 0)), mode='constant')
            _a.append(a)
        _attentions.append(_a)
    return _attentions


def binet_model_fn(dataset,
                   latent_dim=None,
                   use_attributes=None,
                   use_present_activity=None,
                   use_present_attributes=None,
                   encode=None,
                   decode=None,
                   use_attention=None,
                   sparse=False,
                   postfix=''):
    from keras.models import Model
    from keras.layers import Input
    from keras.layers import Embedding
    from keras.layers import concatenate
    from keras.layers import GRU
    from keras.layers import BatchNormalization
    from keras.layers import Dense
    from keras.optimizers import Adam
    from april.anomalydetection.binet.attention import Attention

    # Check for compatibility
    if use_attributes and dataset.num_attributes == 1:
        use_attributes = False

    if use_present_attributes and dataset.num_attributes == 2:
        use_present_attributes = False
        use_present_activity = True

    if sparse:
        targets = dataset.train_targets
        loss = 'sparse_categorical_crossentropy'
    else:
        targets = dataset.onehot_train_targets
        loss = 'categorical_crossentropy'

    if not use_attributes:
        features = dataset.features[:1]
        targets = targets[:1]
    else:
        features = dataset.features

    if latent_dim is None:
        latent_dim = min(int(dataset.max_len * 2), 64)  # clipping at 64 was not part of original paper

    # Build inputs (and encoders if enabled) for past events
    embeddings = []
    inputs = []
    past_outputs = []
    for feature, attr_dim, attr_key in zip(features, dataset.attribute_dims, dataset.attribute_keys):
        i = Input(shape=(None,), name=f'past_{attr_key}{postfix}')
        inputs.append(i)

        voc_size = int(attr_dim + 1)  # we start at 1, hence plus 1
        emb_size = np.clip(int(voc_size / 10), 2, 16)
        embedding = Embedding(input_dim=voc_size,
                              output_dim=emb_size,
                              input_length=feature.shape[1],
                              mask_zero=True)
        embeddings.append(embedding)

        x = embedding(i)

        if encode:
            x, _ = GRU(latent_dim,
                       return_sequences=True,
                       return_state=True,
                       name=f'past_encoder_{attr_key}{postfix}')(x)
            x = BatchNormalization()(x)

        past_outputs.append(x)

    # Build inputs (and encoders if enabled) for present event
    present_features = []
    present_outputs = []
    if use_attributes and (use_present_activity or use_present_attributes):
        # Generate present features, by skipping the first event and adding one padding event at the end
        present_features = [np.pad(f[:, 1:], ((0, 0), (0, 1)), 'constant') for f in features]

        if use_attributes and not use_present_attributes:
            # Use only the activity features
            present_features = present_features[:1]

        for feature, embedding, attr_key in zip(present_features, embeddings, dataset.attribute_keys):
            i = Input(shape=(None,), name=f'present_{attr_key}{postfix}')
            inputs.append(i)

            x = embedding(i)

            if encode:
                x = GRU(latent_dim,
                        return_sequences=True,
                        name=f'present_encoder_{attr_key}{postfix}')(x)
                x = BatchNormalization()(x)

            present_outputs.append(x)

    # Build output layers for each attribute to predict
    outputs = []
    for feature, attr_dim, attr_key in zip(features, dataset.attribute_dims, dataset.attribute_keys):
        if attr_key == 'name' or not use_attributes or (not use_present_activity and not use_present_attributes):
            x = past_outputs
        # Else predict the attribute
        else:
            x = present_outputs[:1]
            if use_present_attributes:
                for past_o, present_o, at_key in zip(past_outputs[1:], present_outputs[1:], dataset.attribute_keys[1:]):
                    if attr_key == at_key:
                        x.append(past_o)
                    else:
                        x.append(present_o)
            else:
                x += past_outputs[1:]

        if use_attention:
            attentions = []
            for _x, at_key in zip(x, dataset.attribute_keys):
                a, _ = Attention(return_sequences=True,
                                 return_coefficients=True,
                                 name=f'attention_{attr_key}/{at_key}{postfix}')(_x)
                attentions.append(a)
            x = attentions
        if len(x) > 1:
            x = concatenate(x)
        else:
            x = x[0]

        if decode:
            x = GRU(latent_dim,
                    return_sequences=True,
                    name=f'decoder_{attr_key}{postfix}')(x)
            x = BatchNormalization()(x)

        o = Dense(int(attr_dim), activation='softmax', name=f'out_{attr_key}{postfix}')(x)
        outputs.append(o)

    # Combine features and build model
    features = features + present_features
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=Adam(),
        loss=loss
    )

    return model, features, targets
