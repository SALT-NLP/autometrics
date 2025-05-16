from autometrics.metrics.Metric import Metric

class MultiMetric(Metric):
    """
    Abstract class for metrics that return multiple values
    """
    def __init__(self, name, description, submetric_names=[], **kwargs) -> None:
        super().__init__(name, description, **kwargs)
        self.submetric_names = submetric_names

    def _calculate_impl(self, input, output, references=None, **kwargs):
        """
        Actual implementation of the metric calculation
        """
        pass

    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        """
        Calculate the metric for a batch of inputs and outputs. The default implementation simply calls _calculate_impl for each input/output pair.
        Override this method if you can calculate the metric more efficiently for a batch of inputs/outputs.
        """
        if references is None:
            references = [None] * len(inputs)

        results = []
        for i, o, r in zip(inputs, outputs, references):
            results.append(self._calculate_impl(i, o, r, **kwargs))

        # Swap the indices so that each submetric has its own list
        results = list(zip(*results))

        return results

    def predict(self, dataset, update_dataset=True, **kwargs):
        """
        Calculate the metric for the dataset
        """
        df = dataset.get_dataframe()
        
        # Use the appropriate method based on metric type
        if hasattr(self, 'calculate_row'):
            # For specific implementations (ReferenceBasedMultiMetric, etc.)
            results_list = []
            for _, row in df.iterrows():
                results = self.calculate_row(row, dataset, False, **kwargs)
                results_list.append(results)
            
            # Transpose results to get one list per submetric
            results_by_submetric = list(zip(*results_list))
            
            if update_dataset:
                for i, submetric_name in enumerate(self.submetric_names):
                    df[submetric_name] = results_by_submetric[i]
                    if submetric_name not in dataset.get_metric_columns():
                        dataset.get_metric_columns().append(submetric_name)
                
                dataset.set_dataframe(df)
            
            return results_list
        else:
            # Generic implementation
            input_column = dataset.get_input_column()
            output_column = dataset.get_output_column()
            
            inputs = df[input_column].values.tolist()
            outputs = df[output_column].values.tolist()
            
            # Determine if we need references
            references = None
            reference_columns = dataset.get_reference_columns()
            if reference_columns:
                references = df[reference_columns].values.tolist()
            
            results = self.calculate_batched(inputs, outputs, references, **kwargs)
            
            if update_dataset:
                for i, submetric_name in enumerate(self.submetric_names):
                    df[submetric_name] = results[i]
                    if submetric_name not in dataset.get_metric_columns():
                        dataset.get_metric_columns().append(submetric_name)
                
                dataset.set_dataframe(df)
            
            return results

    def get_submetric_names(self):
        return self.submetric_names