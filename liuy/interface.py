
from abc import ABCMeta, abstractmethod

class BaseDeepModel(metaclass=ABCMeta):
    """Base deep model. The model object must inherit form this class."""

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

class BaseDataSlection(metaclass=ABCMeta):
    """Base data selection . The data selection object must inherit form this class."""

    def __init__(self, source_data_dir, target_data_dir, **kwargs):
        """
        :param source_data_dir: the root of the datasetto be selected
        :param target_data_dir: the root of the datasetto has been selected
        """
        self.source_data_dir = source_data_dir
        self.target_data_dir = target_data_dir

    @abstractmethod
    def select(self, threshold, **kwargs):
        """select the data from source_data_dir and save to target_data_dir .

        """


