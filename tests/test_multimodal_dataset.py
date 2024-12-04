import os

import numpy as np
import pytest
from pythae.data.datasets import DatasetOutput

from multivae.data.datasets.base import MultimodalBaseDataset


class Test:
    @pytest.fixture(params=["no_transform", "transform"])
    def input_dataset_test(self, request):
        data = dict(
            mod1=np.array([[1, 2], [4, 5]]),
            mod2=np.array([[67, 2, 3], [1, 2, 3]]),
        )
        labels = np.array([0, 1])
        transform = None
        if request.param == "transform":
            transform = dict(
                mod1=lambda x: x,
                mod2=lambda x: -x,
            )
        return dict(data=data, labels=labels, transform=transform)

    def test_create_dataset(self, input_dataset_test):
        dataset = MultimodalBaseDataset(**input_dataset_test)

        sample = dataset[0]
        assert type(sample) == DatasetOutput
        assert np.all(sample["data"]["mod1"] == np.array([1, 2]))
        if dataset.transform is None:
            assert np.all(sample["data"]["mod2"] == np.array([67, 2, 3]))
        else:
            assert np.all(sample["data"]["mod2"] == -np.array([67, 2, 3]))
        assert sample["labels"] == 0
