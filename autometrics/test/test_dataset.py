import pytest
import pandas as pd
import numpy as np
import warnings
from autometrics.dataset.Dataset import Dataset
from autometrics.dataset.PairwiseDataset import PairwiseDataset

@pytest.fixture
def sample_data():
    data = {
        'id': [1, 2, 3, 4, 5],
        'input': ['Input 1', 'Input 2', 'Input 3', 'Input 4', 'Input 5'],
        'output': ['Output 1', 'Output 2', 'Output 3', 'Output 4', 'Output 5']
    }
    return pd.DataFrame(data)

@pytest.fixture
def basic_dataset(sample_data):
    return Dataset(
        dataframe=sample_data,
        target_columns=[],
        ignore_columns=[],
        metric_columns=[],
        name="test_dataset",
        data_id_column='id',
        input_column='input',
        output_column='output',
        reference_columns=None
    )

@pytest.fixture
def pairwise_data():
    data = {
        'id': [1, 2, 3, 4, 5],
        'model_id_1': ['A'] * 5,
        'model_id_2': ['B'] * 5,
        'input': ['Input 1', 'Input 2', 'Input 3', 'Input 4', 'Input 5'],
        'output1': ['Output A1', 'Output A2', 'Output A3', 'Output A4', 'Output A5'],
        'output2': ['Output B1', 'Output B2', 'Output B3', 'Output B4', 'Output B5']
    }
    return pd.DataFrame(data)

@pytest.fixture
def pairwise_dataset(pairwise_data):
    return PairwiseDataset(
        dataframe=pairwise_data,
        target_columns=[],
        ignore_columns=[],
        metric_columns=[],
        name="test_pairwise_dataset",
        data_id_column='id',
        model_id_column_1='model_id_1',
        model_id_column_2='model_id_2',
        input_column='input',
        output_column_1='output1',
        output_column_2='output2',
        reference_columns=None
    )

def test_get_subset_normal(basic_dataset):
    """Test get_subset with a size smaller than the available data"""
    subset = basic_dataset.get_subset(3, seed=42)
    
    # Verify subset size
    assert len(subset.get_dataframe()) == 3
    
    # Verify subset maintains properties of original dataset
    assert subset.get_input_column() == 'input'
    assert subset.get_output_column() == 'output'

def test_get_subset_exact_size(basic_dataset):
    """Test get_subset with a size equal to the available data"""
    subset = basic_dataset.get_subset(5, seed=42)
    
    # Should get all rows
    assert len(subset.get_dataframe()) == 5
    
    # Should still have all the original data (but possibly in different order)
    assert set(subset.get_dataframe()['id']) == set(range(1, 6))

def test_get_subset_oversized(basic_dataset):
    """Test get_subset with a size larger than the available data"""
    # This should issue a warning but not fail
    with pytest.warns() as recorded_warnings:
        subset = basic_dataset.get_subset(10, seed=42)
        
        # Verify a warning was issued
        assert any("Requested subset size 10 is larger than available data" in str(warning.message) for warning in recorded_warnings)
    
    # Should get all available rows
    assert len(subset.get_dataframe()) == 5
    
    # Should have all the original data
    assert set(subset.get_dataframe()['id']) == set(range(1, 6))

def test_splits_with_max_size(basic_dataset):
    """Test that get_splits works with max_size larger than available data"""
    # This should not fail even though max_size is larger than available data
    train, val, test = basic_dataset.get_splits(train_ratio=0.6, val_ratio=0.2, seed=42, max_size=100)
    
    # Verify splits were created
    assert isinstance(train, Dataset)
    assert isinstance(val, Dataset)
    assert isinstance(test, Dataset)
    
    # Check that the splits together contain all the original data
    total_rows = len(train.get_dataframe()) + len(val.get_dataframe()) + len(test.get_dataframe())
    assert total_rows == 5

def test_pairwise_splits_with_max_size(pairwise_dataset):
    """Test that PairwiseDataset.get_splits works with max_size larger than available data"""
    # This should not fail even though max_size is larger than available data
    train, val, test = pairwise_dataset.get_splits(train_ratio=0.6, val_ratio=0.2, seed=42, max_size=100)
    
    # Verify splits are PairwiseDataset instances
    assert isinstance(train, PairwiseDataset)
    assert isinstance(val, PairwiseDataset)
    assert isinstance(test, PairwiseDataset)
    
    # Check that the splits together contain all the original data
    total_rows = len(train.get_dataframe()) + len(val.get_dataframe()) + len(test.get_dataframe())
    assert total_rows == 5
    
    # Verify the pairwise-specific properties are preserved
    assert train.get_output_column_1() == 'output1'
    assert train.get_output_column_2() == 'output2' 