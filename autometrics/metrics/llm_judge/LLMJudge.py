import dspy
from autometrics.metrics.Metric import Metric
from autometrics.util.format import get_default_formatter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class LLMAsAJudgeSignature(dspy.Signature):
    """Given an input text, the task description that the model was trying to follow, and a metric to rate the text on, return a score from 1-5 on this metric."""
    text = dspy.InputField(desc="The input text that we want to rate.")
    task_description = dspy.InputField(desc="A description of the task that the model was trying to solve when it generated the text.  Could be left blank if not available.")
    metric = dspy.InputField(desc="The metric that we want to rate the text on.")
    score = dspy.OutputField(desc="The score that the text should recieve on this metric (1=low, 5=high).")

class LLMAsAJudge(dspy.Module):
    def __init__(self):
        super(LLMAsAJudge, self).__init__()
        self.generate_score = dspy.ChainOfThought(LLMAsAJudgeSignature)

    def forward(self, text, metric, task_description=None):
        if task_description is None:
            task_description = "None"
        score = self.generate_score(task_description=task_description, text=text, metric=metric).score
        # Convert the string score to a float by stripping any additional text and converting to a float
        if '\n' in score:
            score = score.split('\n')[0]

        if '.' in score:
            score = score.split('.')[0]
            
        score = float(score.strip())

        return dspy.Prediction(text=text, metric=metric, score=score)
    
def grade_row(row, axis, llm, formatter, task_description):
    '''Helper function to grade a single row'''
    with dspy.settings.context(lm=llm):
        return LLMAsAJudge()(formatter(row), axis, task_description).score
    
class LLMJudge(Metric):
    def __init__(self, name, description, model, dataset, evaluation_axis, formatter=None, task_description=None):
        super().__init__(name, description)
        self.model = model
        if formatter is None:
            self.formatter = get_default_formatter(dataset)
        else:
            self.formatter = formatter
        self.dataset = dataset
        self.task_description = task_description
        self.evaluation_axis = evaluation_axis
        
    def calculate(self, input, output, references=None, **kwargs):
        row = {self.dataset.get_input_column(): input, self.dataset.get_output_column(): output}
        if references is not None:
            for i, ref in enumerate(references):
                row[self.dataset.get_reference_columns()[i]] = ref

        grade_row(row, self.evaluation_axis, self.model, self.formatter, self.task_description)
    
    def predict(self, dataset, update_dataset=True, max_workers=64, metric_name=None, **kwargs):
            '''
                Grade the dataframe using the LLM judge in parallel with progress bar
            '''
            if metric_name is None:
                metric_name = self.evaluation_axis.split(":")[0].replace("*", "") + "_" + self.model.kwargs['model'].split("/")[-1]

            df = dataset.get_dataframe()

            results = []

            # Create a ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit tasks to the executor
                futures = {executor.submit(grade_row, row, self.evaluation_axis, self.model, self.formatter, self.task_description): index for index, row in df.iterrows()}

                # Collect the results with tqdm progress bar
                for future in tqdm(as_completed(futures), total=len(futures), desc="Grading rows", unit="row"):
                    index = futures[future]
                    try:
                        score = future.result()
                        if update_dataset:
                            df.at[index, metric_name] = score
                        results.append((index, score))
                    except Exception as e:
                        print(f"Error processing row {index}: {e}")

            if update_dataset:
                dataset.set_dataframe(df)
                if metric_name not in dataset.get_metric_columns():
                    dataset.get_metric_columns().append(metric_name)

            results.sort(key=lambda x: x[0])
            return [score for _, score in results]