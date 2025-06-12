from autometrics.dataset.Dataset import Dataset
import pandas as pd
from typing import Literal
from autometrics.metrics.dummy import DummyMetric
class CoGym(Dataset):
    def __init__(self, path='./autometrics/dataset/datasets/cogym/cogym.csv', task_type: str = "travel_planning", eval_type: Literal['process', 'outcome'] = 'process'): # Path to the dataset: './autometrics/dataset/datasets/cogym/cogym.csv'
        # userId,sessionId,modelName,task,query,createdAt,finishedAt,agentRating,communicationRating,outcomeRating,agentFeedback,finished,bookmarked,agentType,conversation,formatted_conversation,outcome
        df = pd.read_csv(path)

        df = df[df['task'] == task_type]
        df.drop(columns=['createdAt', 'finishedAt', 'bookmarked', 'agentType'], inplace=True)

        if eval_type == 'process':
            target_columns = ['agentRating', 'communicationRating']
        else:
            target_columns = ['outcomeRating']

        ignore_columns = ["agentFeedback", "outcome", "userId", "sessionId", "modelName", "task", "query", "formatted_conversation", "conversation"]
        metric_columns = []

        name = "cogym"

        data_id_column = "sessionId"
        model_id_column = "modelName"
        input_column = "query"
        output_column = "formatted_conversation" if eval_type == 'process' else "outcome"
        reference_columns = []

        metrics = [DummyMetric(col) for col in metric_columns]

        super().__init__(df, target_columns, ignore_columns, metric_columns, name, data_id_column, model_id_column, input_column, output_column, reference_columns, metrics)

class CoGymTravelOutcome(CoGym):
    def __init__(self):
        super().__init__('./autometrics/dataset/datasets/cogym/cogym.csv', 'travel_planning', 'outcome')
        self.name = "cogym_travel_outcome"
        self.task_description = """You are a proficient travel planner. Your task is to plan a detailed travel itinerary for a user or a group of users. Note that all the information in your plan should be grounded in searched information. The plan should also align with commonsense.\nHere is the initial query: \nEdit the travel plan for the query in the editor. Note that the editor is in a shared workbench which means it can be viewed and edited by every team member. Your change to the editor will also be shown to the team member in real time so you don't need to send the content to them. Once finished, you performance will be evaluated based on the feasibility and user satisfaction of the plan in the editor.\nIt's a good practice to include links you found useful in the plan."""

class CoGymTravelProcess(CoGym):
    def __init__(self):
        super().__init__('./autometrics/dataset/datasets/cogym/cogym.csv', 'travel_planning', 'process')
        self.name = "cogym_travel_process"
        self.task_description = """You are a proficient travel planner. Your task is to plan a detailed travel itinerary for a user or a group of users. Note that all the information in your plan should be grounded in searched information. The plan should also align with commonsense.\nHere is the initial query: \nEdit the travel plan for the query in the editor. Note that the editor is in a shared workbench which means it can be viewed and edited by every team member. Your change to the editor will also be shown to the team member in real time so you don't need to send the content to them. Once finished, you performance will be evaluated based on the feasibility and user satisfaction of the plan in the editor.\nIt's a good practice to include links you found useful in the plan."""

class CoGymTabularOutcome(CoGym):
    def __init__(self):
        super().__init__('./autometrics/dataset/datasets/cogym/cogym.csv', 'tabular_analysis', 'outcome')
        self.name = "cogym_tabular_outcome"
        self.task_description = """Your task is to analyze the provided tabular dataset(s)\nSpecifically, here is the user query that you need to follow:\n{user_query}"""
        
class CoGymTabularProcess(CoGym):
    def __init__(self):
        super().__init__('./autometrics/dataset/datasets/cogym/cogym.csv', 'tabular_analysis', 'process')
        self.name = "cogym_tabular_process"
        self.task_description = """Your task is to analyze the provided tabular dataset(s)\nSpecifically, here is the user query that you need to follow:\n{user_query}"""
        
class CoGymLessonOutcome(CoGym):
    def __init__(self):
        super().__init__('./autometrics/dataset/datasets/cogym/cogym.csv', 'lesson_planning', 'outcome')
        self.name = "cogym_lesson_outcome"
        self.task_description = """Your task is to create a lesson plan based on the given curriculum and query. Write the plan in the editor.\nPedagogical Knowledge: A high-quality lesson plan includes:\n1. Integrating objectives for student learning.\n2. Incorporating teaching/learning activities.\n3. Embedding strategies to check student understanding.\nSome teaching strategies include:\n1. Gaining attention: Techniques to capture students' interest.\n2. Informing learners of objectives: Clearly stating what students will be able to do by the end of the lesson.\n3. Stimulating recall of prior learning: Helping students connect new information with what they already know.\n4. Presenting the content: Delivering the new information in various engaging ways.\n5. Providing learning guidance: Offering support and strategies to aid learning processes.\n6. Eliciting performance: Encouraging students to demonstrate their understanding.\n7. Providing feedback: Giving constructive feedback to enhance learning.\n8. Assessing performance: Measuring students' understanding to ensure learning objectives are met.\n9. Enhancing retention and transfer: Helping students apply what they learned to different contexts.\nAlso, please add links to resources that can be used in the lesson plan if any.\n\nQuery: {user_query}"""

class CoGymLessonProcess(CoGym):
    def __init__(self):
        super().__init__('./autometrics/dataset/datasets/cogym/cogym.csv', 'lesson_planning', 'process')
        self.name = "cogym_lesson_process"
        self.task_description = """Your task is to create a lesson plan based on the given curriculum and query. Write the plan in the editor.\nPedagogical Knowledge: A high-quality lesson plan includes:\n1. Integrating objectives for student learning.\n2. Incorporating teaching/learning activities.\n3. Embedding strategies to check student understanding.\nSome teaching strategies include:\n1. Gaining attention: Techniques to capture students' interest.\n2. Informing learners of objectives: Clearly stating what students will be able to do by the end of the lesson.\n3. Stimulating recall of prior learning: Helping students connect new information with what they already know.\n4. Presenting the content: Delivering the new information in various engaging ways.\n5. Providing learning guidance: Offering support and strategies to aid learning processes.\n6. Eliciting performance: Encouraging students to demonstrate their understanding.\n7. Providing feedback: Giving constructive feedback to enhance learning.\n8. Assessing performance: Measuring students' understanding to ensure learning objectives are met.\n9. Enhancing retention and transfer: Helping students apply what they learned to different contexts.\nAlso, please add links to resources that can be used in the lesson plan if any.\n\nQuery: {user_query}"""
        
        
        
        