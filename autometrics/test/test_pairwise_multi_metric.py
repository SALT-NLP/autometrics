import unittest
import pandas as pd
import numpy as np
from autometrics.metrics.PairwiseMetric import PairwiseMetric
from autometrics.metrics.PairwiseMultiMetric import PairwiseMultiMetric
from autometrics.metrics.Metric import Metric
from autometrics.metrics.MultiMetric import MultiMetric
from autometrics.dataset.Dataset import Dataset
from autometrics.dataset.PairwiseDataset import PairwiseDataset

# Mock MultiMetric for testing
class MockMultiMetric(MultiMetric):
    """A mock MultiMetric that returns multiple scores"""
    
    def __init__(self):
        super().__init__(
            name="mock_multi_metric",
            description="A mock metric that returns multiple scores",
            submetric_names=["length", "word_count"]
        )
    
    def _calculate_impl(self, input, output, references=None, **kwargs):
        """Return length and word count as two separate metrics"""
        length = len(output)
        word_count = len(output.split()) if isinstance(output, str) else 0
        return [length, word_count]
    
    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        """Return length and word count for each input in batch form"""
        # Call _calculate_impl for each input and output
        results = []
        for i, o in zip(inputs, outputs):
            results.append(self._calculate_impl(i, o, None, **kwargs))
            
        # Transpose the results to get the format expected by MultiMetric:
        # [[length_1, length_2, ...], [word_count_1, word_count_2, ...]]
        lengths, word_counts = zip(*results)
        return [list(lengths), list(word_counts)]

class TestPairwiseMultiMetric(unittest.TestCase):
    def setUp(self):
        # Create a mock multi-metric
        self.multi_metric = MockMultiMetric()
        
        # Create a pairwise wrapper for it using PairwiseMultiMetric directly
        self.pairwise_metric = PairwiseMultiMetric(multi_metric=self.multi_metric)
        
        # Verify it's a PairwiseMultiMetric
        self.assertIsInstance(self.pairwise_metric, PairwiseMultiMetric)
        
        # Create a dataset for testing
        self.data = {
            'id': [1, 2, 3],
            'model_id_1': ['model_A', 'model_A', 'model_A'],
            'model_id_2': ['model_B', 'model_B', 'model_B'],
            'input': ['Question 1', 'Question 2', 'Question 3'],
            'output1': ['Short', 'Medium length text', 'Very long response indeed'],
            'output2': ['Longer response', 'Tiny', 'Medium']
        }
        self.df = pd.DataFrame(self.data)
        
        # Create a pairwise dataset
        self.pairwise_dataset = PairwiseDataset(
            dataframe=self.df,
            target_columns=[],
            ignore_columns=[],
            metric_columns=[],
            name="test_dataset",
            data_id_column='id',
            model_id_column_1='model_id_1',
            model_id_column_2='model_id_2',
            input_column='input',
            output_column_1='output1',
            output_column_2='output2',
            reference_columns=None
        )

    def test_submetric_names(self):
        """Test that PairwiseMultiMetric correctly prefixes submetric names"""
        expected_names = ['pairwise_length', 'pairwise_word_count']
        self.assertEqual(self.pairwise_metric.get_submetric_names(), expected_names)

    def test_calculate(self):
        """Test that calculate returns multiple values"""
        input_text = "Test input"
        output_1 = "Short text"  # len=10, word_count=2
        output_2 = "Longer response"  # len=15, word_count=2
        
        result = self.pairwise_metric.calculate(input_text, output_1, output_2)
        
        # Should return [length1-length2, word_count1-word_count2]
        expected = [10-15, 2-2]  # [-5, 0]
        self.assertEqual(result, expected)

    def test_calculate_batched(self):
        """Test that calculate_batched returns multiple values for each input"""
        inputs = ["Input 1", "Input 2", "Input 3"]
        outputs_1 = ["Short text", "Medium length", "One two three"]
        outputs_2 = ["Longer response", "Tiny", "Word"]
        
        results = self.pairwise_metric.calculate_batched(inputs, outputs_1, outputs_2)
        
        # Each result should be a list of differences
        self.assertEqual(len(results), 3)
        self.assertIsInstance(results[0], list)
        
        # Verify the differences
        # "Short text" (10, 2) vs "Longer response" (15, 2) = [-5, 0]
        # "Medium length" (13, 2) vs "Tiny" (4, 1) = [9, 1]
        # "One two three" (13, 3) vs "Word" (4, 1) = [9, 2]
        expected = [
            [10-15, 2-2],  # [-5, 0]
            [13-4, 2-1],   # [9, 1]
            [13-4, 3-1]    # [9, 2]
        ]
        self.assertEqual(results, expected)

    def test_with_pairwise_dataset(self):
        """Test integration with PairwiseDataset"""
        # Add the metric to the dataset
        self.pairwise_dataset.add_metric(self.pairwise_metric)
        
        # Get the dataframe
        df = self.pairwise_dataset.get_dataframe()
        
        # Verify the columns were added
        self.assertIn("pairwise_length", df.columns)
        self.assertIn("pairwise_word_count", df.columns)
        
        # Verify the values are correct
        # "Short" (5, 1) vs "Longer response" (15, 2) = [-10, -1]
        # "Medium length text" (18, 3) vs "Tiny" (4, 1) = [14, 2]
        # "Very long response indeed" (25, 4) vs "Medium" (6, 1) = [19, 3]
        expected_length = [-10, 14, 19]
        expected_word_count = [-1, 2, 3]
        
        self.assertEqual(df["pairwise_length"].tolist(), expected_length)
        self.assertEqual(df["pairwise_word_count"].tolist(), expected_word_count)

    def test_auto_wrapped_multi_metric(self):
        """Test that MultiMetric is automatically wrapped in PairwiseMultiMetric"""
        # Create a fresh dataset
        dataset = PairwiseDataset(
            dataframe=self.df.copy(),
            target_columns=[],
            ignore_columns=[],
            metric_columns=[],
            name="test_dataset",
            data_id_column='id',
            input_column='input',
            output_column_1='output1',
            output_column_2='output2'
        )
        
        # Add the MultiMetric directly (should be auto-wrapped)
        multi_metric = MockMultiMetric()
        dataset.add_metric(multi_metric)
        
        # Get the dataframe
        df = dataset.get_dataframe()
        
        # Verify the columns were added with the pairwise prefix
        self.assertIn("pairwise_length", df.columns)
        self.assertIn("pairwise_word_count", df.columns)
        
        # Get metric values should return a dataframe with both columns
        results = dataset.get_metric_values(multi_metric)
        self.assertEqual(list(results.columns), ["pairwise_length", "pairwise_word_count"])

    def test_get_metric_values(self):
        """Test that get_metric_values returns a dataframe with all submetric columns"""
        # Add the metric to the dataset
        self.pairwise_dataset.add_metric(self.pairwise_metric)
        
        # Get metric values
        results = self.pairwise_dataset.get_metric_values(self.pairwise_metric)
        
        # Should return a dataframe with both columns
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(list(results.columns), ["pairwise_length", "pairwise_word_count"])

if __name__ == '__main__':
    unittest.main() 