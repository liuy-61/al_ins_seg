
from abc import ABCMeta, abstractmethod

class BaseInsSegModel(metaclass=ABCMeta):
    """Base  model. The model object must inherit form this class."""

    def __init__(self, project_id, data_dir, **kwargs):
        """
                :param project_id: The project id that use this model.
                :param train_dir: the dataset for training .
                :param test_dir: the dataset for testing
                :param
                :param
                :param kwargs: Other necessary params.

        """
        self._proj_id = project_id
        self.data_dir = data_dir

    @abstractmethod
    def fit(self, **kwargs):
        """Train the model use the all train  dataset.
        """
    def fit_on_subset(self, **kwargs):
        """Train the model use the subset train  dataset.
        """

    @abstractmethod
    def predict_proba(self, data_dir, **kwargs):
        """proba predict.

        :param data_dir: str
            The path to the data folder.

        :param kwargs: dict
            Other necessary params.
        """

    @abstractmethod
    def predict(self, data_dir, **kwargs):
        """predict class label.

        :param data_dir: str
            The path to the data folder.
        """

    @abstractmethod
    def test(self, data_dir, label, batch_size:'int', **kwargs):
        """tets the model.
        """

    @abstractmethod
    def save_model(self):
        """Save the model after using (distributed system)."""

class BaseSampler(metaclass=ABCMeta):
    def __init__(self, data_loader, **kwargs):
        self.data_loader = data_loader

    def select_batch(self, N, already_selct, **kwargs):
        """

        :param N: batch size
        :param already_selct: index of datapoints already selected
        :param kwargs:
        :return: index of  points selected
        """
        return

class BaseAl(metaclass=ABCMeta):
    def __init__(self, seg_model, sampler, **kwargs):
        """

        :param seg_model:  model used to score the samplers.  Expects fit and predict
        methods to be implemented.
        :param sampler: sampling class from sampling_methods, assumes reference
        passed in and sampler not yet instantiated.
        :param kwargs:
        """
        self.seg_model = seg_model
        self.sampler = sampler

    def select_batch(self, N, already_selected,
                     **kwargs):
        pass




