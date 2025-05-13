import unittest
import pandas as pd
from autometrics.metrics.PairwiseMetric import PairwiseMetric
from autometrics.metrics.Metric import Metric
from autometrics.dataset.Dataset import Dataset
from autometrics.dataset.PairwiseDataset import PairwiseDataset

# Simple mock metric for testing
class MockMetric(Metric):
    def __init__(self):
        super().__init__("mock_metric", "A mock metric that returns the length of the output")

    def _calculate_impl(self, input, output, references=None, **kwargs):
        return len(output)

    def predict(self, dataset, update_dataset=True, **kwargs):
        # Simplified implementation for testing
        df = dataset.get_dataframe()
        output_column = dataset.get_output_column()
        results = [len(out) for out in df[output_column]]
        
        if update_dataset:
            df[self.name] = results
            dataset.set_dataframe(df)
            
            if self.name not in dataset.get_metric_columns():
                dataset.get_metric_columns().append(self.name)
                
        return results

# Direct pairwise metric without wrapping another metric
class DirectPairwiseMetric(PairwiseMetric):
    def __init__(self):
        super().__init__(scalar_metric=None, 
                         name="direct_pairwise", 
                         description="A metric that directly compares two outputs")
    
    def _calculate_pairwise_impl(self, input, output_1, output_2, references=None, **kwargs):
        # Simple implementation: return difference in lengths
        return len(output_1) - len(output_2)

class TestPairwiseMetric(unittest.TestCase):
    def setUp(self):
        # Create a mock metric
        self.mock_metric = MockMetric()
        # Create a pairwise metric that wraps the mock metric
        self.pairwise_metric = PairwiseMetric(self.mock_metric)
        # Create a direct pairwise metric
        self.direct_metric = DirectPairwiseMetric()
        
        # Create a dataset for testing
        self.data = {
            'id': [1, 2, 3],
            'model_id_1': ['model_A', 'model_A', 'model_A'],
            'model_id_2': ['model_B', 'model_B', 'model_B'],
            'input': ['Question 1', 'Question 2', 'Question 3'],
            'output1': ['Short', 'Medium length', 'Very long response indeed'],
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

    def test_initialization(self):
        """Test that the PairwiseMetric initializes correctly"""
        self.assertEqual(self.pairwise_metric.metric, self.mock_metric)
        self.assertEqual(self.pairwise_metric.get_name(), "pairwise_mock_metric")
        self.assertEqual(self.pairwise_metric.get_description(), "Pairwise comparison using A mock metric that returns the length of the output")
        
        # Test direct pairwise metric
        self.assertIsNone(self.direct_metric.metric)
        self.assertEqual(self.direct_metric.get_name(), "direct_pairwise")
        self.assertEqual(self.direct_metric.get_description(), "A metric that directly compares two outputs")

    def test_direct_pairwise_requires_name(self):
        """Test that direct pairwise metrics require name and description"""
        with self.assertRaises(ValueError):
            PairwiseMetric(scalar_metric=None)

    def test_custom_naming(self):
        """Test custom naming"""
        custom_metric = PairwiseMetric(self.mock_metric, name="custom_name", description="custom description")
        self.assertEqual(custom_metric.get_name(), "custom_name")
        self.assertEqual(custom_metric.get_description(), "custom description")

    def test_calculate(self):
        """Test the calculate method"""
        # Create inputs and outputs for testing
        input_text = "Test input"
        output_1 = "Short"  # len = 5
        output_2 = "Longer output"  # len = 13
        
        # Test wrapped metric
        result = self.pairwise_metric.calculate(input_text, output_1, output_2)
        self.assertEqual(result, 5 - 13)
        
        # Test direct pairwise metric
        result = self.direct_metric.calculate(input_text, output_1, output_2)
        self.assertEqual(result, 5 - 13)  # Same result, different implementation

    def test_calculate_batched(self):
        """Test the calculate_batched method"""
        inputs = ["Input 1", "Input 2", "Input 3"]
        outputs_1 = ["Short", "Medium text", "Equal"]
        outputs_2 = ["Longer output", "Tiny", "Equal"]
        
        # Expected results
        expected = [
            len("Short") - len("Longer output"),       # 5 - 13 = -8
            len("Medium text") - len("Tiny"),          # 11 - 4 = 7
            len("Equal") - len("Equal")                # 5 - 5 = 0
        ]
        
        # Test wrapped metric
        results = self.pairwise_metric.calculate_batched(inputs, outputs_1, outputs_2)
        self.assertEqual(results, expected)
        
        # Test direct pairwise metric
        results = self.direct_metric.calculate_batched(inputs, outputs_1, outputs_2)
        self.assertEqual(results, expected)  # Same result, different implementation

    def test_with_pairwise_dataset(self):
        """Test integration with a PairwiseDataset"""
        # Add metrics to the dataset
        self.pairwise_dataset.add_metric(self.pairwise_metric)
        
        # Check that the metric was added and calculated
        df = self.pairwise_dataset.get_dataframe()
        self.assertIn("pairwise_mock_metric", df.columns)
        
        # Verify results
        expected = [
            len('Short') - len('Longer response'),               # 5 - 15 = -10
            len('Medium length') - len('Tiny'),                  # 13 - 4 = 9
            len('Very long response indeed') - len('Medium')     # 25 - 6 = 19
        ]
        
        results = df["pairwise_mock_metric"].tolist()
        self.assertEqual(results, expected)
        
        # Test with direct pairwise metric
        self.pairwise_dataset.add_metric(self.direct_metric)
        df = self.pairwise_dataset.get_dataframe()
        self.assertIn("direct_pairwise", df.columns)
        
        # Should have the same results as they implement the same logic
        results = df["direct_pairwise"].tolist()
        self.assertEqual(results, expected)

    def test_auto_wrapping_metric(self):
        """Test that regular metrics get automatically wrapped as PairwiseMetric"""
        # Create a new dataset
        pairwise_dataset = PairwiseDataset(
            dataframe=self.df.copy(),
            target_columns=[],
            ignore_columns=[],
            metric_columns=[],
            name="test_dataset",
            data_id_column='id',
            model_id_column_1='model_id_1',
            model_id_column_2='model_id_2',
            input_column='input',
            output_column_1='output1',
            output_column_2='output2'
        )
        
        # Add a regular Metric (not PairwiseMetric)
        mock_metric = MockMetric()
        pairwise_dataset.add_metric(mock_metric)
        
        # It should be wrapped as PairwiseMetric
        df = pairwise_dataset.get_dataframe()
        self.assertIn("pairwise_mock_metric", df.columns)
        
        # Results should match our expected differences
        expected = [
            len('Short') - len('Longer response'),
            len('Medium length') - len('Tiny'),
            len('Very long response indeed') - len('Medium')
        ]
        
        results = df["pairwise_mock_metric"].tolist()
        self.assertEqual(results, expected)

    def test_dataset_getters(self):
        """Test the PairwiseDataset getters"""
        self.assertEqual(self.pairwise_dataset.get_output_column_1(), 'output1')
        self.assertEqual(self.pairwise_dataset.get_output_column_2(), 'output2')
        self.assertEqual(self.pairwise_dataset.get_model_id_column_1(), 'model_id_1')
        self.assertEqual(self.pairwise_dataset.get_model_id_column_2(), 'model_id_2')

    def test_dataset_splitting(self):
        """Test dataset splitting and copying preserves pairwise nature"""
        # Split the dataset
        train, val, test = self.pairwise_dataset.get_splits(train_ratio=0.34, val_ratio=0.33, seed=42)
        
        # Check that each split is a PairwiseDataset
        self.assertIsInstance(train, PairwiseDataset)
        self.assertIsInstance(val, PairwiseDataset)
        self.assertIsInstance(test, PairwiseDataset)
        
        # Check that each split has the necessary columns and attributes
        for dataset in [train, val, test]:
            self.assertEqual(dataset.get_output_column_1(), 'output1')
            self.assertEqual(dataset.get_output_column_2(), 'output2')
            self.assertEqual(dataset.get_model_id_column_1(), 'model_id_1')
            self.assertEqual(dataset.get_model_id_column_2(), 'model_id_2')
            self.assertIn('output1', dataset.get_dataframe().columns)
            self.assertIn('output2', dataset.get_dataframe().columns)
            self.assertIn('model_id_1', dataset.get_dataframe().columns)
            self.assertIn('model_id_2', dataset.get_dataframe().columns)
        
        # Test subset
        subset = self.pairwise_dataset.get_subset(2, seed=42)
        self.assertIsInstance(subset, PairwiseDataset)
        self.assertEqual(len(subset.get_dataframe()), 2)
        self.assertEqual(subset.get_output_column_1(), 'output1')
        self.assertEqual(subset.get_output_column_2(), 'output2')
        self.assertEqual(subset.get_model_id_column_1(), 'model_id_1')
        self.assertEqual(subset.get_model_id_column_2(), 'model_id_2')
        
        # Test copy
        copied = self.pairwise_dataset.copy()
        self.assertIsInstance(copied, PairwiseDataset)
        self.assertEqual(copied.get_output_column_1(), 'output1')
        self.assertEqual(copied.get_output_column_2(), 'output2')
        self.assertEqual(copied.get_model_id_column_1(), 'model_id_1')
        self.assertEqual(copied.get_model_id_column_2(), 'model_id_2')
        
if __name__ == '__main__':
    unittest.main() 