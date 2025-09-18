from autometrics.dataset.Dataset import Dataset
import pandas as pd
from autometrics.metrics.dummy import DummyMetric


class TauBench(Dataset):
    def __init__(self, dir_path: str = './autometrics/dataset/datasets/taubench', train_file: str = 'train_runs.csv', test_file: str = 'test_runs.csv', name: str = 'taubench'):
        # Load provided splits
        train_path = f"{dir_path}/{train_file}"
        test_path = f"{dir_path}/{test_file}"

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # id,split,model,user_query,messages,reward

        # Column roles
        # Identifiers and metadata
        data_id_column = 'id'
        model_id_column = 'model'

        # Text fields
        input_column = 'user_query'
        output_column = 'messages'
        reference_columns = []

        # Targets: review aspect scores, overall recommendation, and acceptance label
        target_columns = [
            'reward',
        ]

        # Columns to ignore for metric computation
        ignore_columns = [
            'id',
            'messages',
            'split',
            'model',
            'user_query',
        ]

        # Optional auxiliary numeric/text columns that could be tracked as metrics if present
        # Exclude targets and ignored columns
        metric_columns = []

        name = name

        # Instantiate metric objects (no-op placeholders by default)
        metrics = [DummyMetric(col) for col in metric_columns]

        # Base dataset will mirror the pattern used elsewhere: use training as the base
        task_description = (
            '# Airline Agent Policy\n\nThe current time is 2024-05-15 15:00:00 EST.\n\nAs an airline agent, you can help users book, modify, or cancel flight reservations.\n\n- Before taking any actions that update the booking database (booking, modifying flights, editing baggage, upgrading cabin class, or updating passenger information), you must list the action details and obtain explicit user confirmation (yes) to proceed.\n\n- You should not provide any information, knowledge, or procedures not provided by the user or available tools, or give subjective recommendations or comments.\n\n- You should only make one tool call at a time, and if you make a tool call, you should not respond to the user simultaneously. If you respond to the user, you should not make a tool call at the same time.\n\n- You should deny user requests that are against this policy.\n\n- You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions.\n\n## Domain Basic\n\n- Each user has a profile containing user id, email, addresses, date of birth, payment methods, reservation numbers, and membership tier.\n\n- Each reservation has an reservation id, user id, trip type (one way, round trip), flights, passengers, payment methods, created time, baggages, and travel insurance information.\n\n- Each flight has a flight number, an origin, destination, scheduled departure and arrival time (local time), and for each date:\n  - If the status is "available", the flight has not taken off, available seats and prices are listed.\n  - If the status is "delayed" or "on time", the flight has not taken off, cannot be booked.\n  - If the status is "flying", the flight has taken off but not landed, cannot be booked.\n\n## Book flight\n\n- The agent must first obtain the user id, then ask for the trip type, origin, destination.\n\n- Passengers: Each reservation can have at most five passengers. The agent needs to collect the first name, last name, and date of birth for each passenger. All passengers must fly the same flights in the same cabin.\n\n- Payment: each reservation can use at most one travel certificate, at most one credit card, and at most three gift cards. The remaining amount of a travel certificate is not refundable. All payment methods must already be in user profile for safety reasons.\n\n- Checked bag allowance: If the booking user is a regular member, 0 free checked bag for each basic economy passenger, 1 free checked bag for each economy passenger, and 2 free checked bags for each business passenger. If the booking user is a silver member, 1 free checked bag for each basic economy passenger, 2 free checked bag for each economy passenger, and 3 free checked bags for each business passenger. If the booking user is a gold member, 2 free checked bag for each basic economy passenger, 3 free checked bag for each economy passenger, and 3 free checked bags for each business passenger. Each extra baggage is 50 dollars.\n\n- Travel insurance: the agent should ask if the user wants to buy the travel insurance, which is 30 dollars per passenger and enables full refund if the user needs to cancel the flight given health or weather reasons.\n\n## Modify flight\n\n- The agent must first obtain the user id and the reservation id.\n\n- Change flights: Basic economy flights cannot be modified. Other reservations can be modified without changing the origin, destination, and trip type. Some flight segments can be kept, but their prices will not be updated based on the current price. The API does not check these for the agent, so the agent must make sure the rules apply before calling the API!\n\n- Change cabin: all reservations, including basic economy, can change cabin without changing the flights. Cabin changes require the user to pay for the difference between their current cabin and the new cabin class. Cabin class must be the same across all the flights in the same reservation; changing cabin for just one flight segment is not possible.\n\n- Change baggage and insurance: The user can add but not remove checked bags. The user cannot add insurance after initial booking.\n\n- Change passengers: The user can modify passengers but cannot modify the number of passengers. This is something that even a human agent cannot assist with.\n\n- Payment: If the flights are changed, the user needs to provide one gift card or credit card for payment or refund method. The agent should ask for the payment or refund method instead.\n\n## Cancel flight\n\n- The agent must first obtain the user id, the reservation id, and the reason for cancellation (change of plan, airline cancelled flight, or other reasons)\n\n- All reservations can be cancelled within 24 hours of booking, or if the airline cancelled the flight. Otherwise, basic economy or economy flights can be cancelled only if travel insurance is bought and the condition is met, and business flights can always be cancelled. The rules are strict regardless of the membership status. The API does not check these for the agent, so the agent must make sure the rules apply before calling the API!\n\n- The agent can only cancel the whole trip that is not flown. If any of the segments are already used, the agent cannot help and transfer is needed.\n\n- The refund will go to original payment methods in 5 to 7 business days.\n\n## Refund\n\n- If the user is silver/gold member or has travel insurance or flies business, and complains about cancelled flights in a reservation, the agent can offer a certificate as a gesture after confirming the facts, with the amount being $100 times the number of passengers.\n\n- If the user is silver/gold member or has travel insurance or flies business, and complains about delayed flights in a reservation and wants to change or cancel the reservation, the agent can offer a certificate as a gesture after confirming the facts and changing or cancelling the reservation, with the amount being $50 times the number of passengers.\n\n- Do not proactively offer these unless the user complains about the situation and explicitly asks for some compensation. Do not compensate if the user is regular member and has no travel insurance and flies (basic) economy.'
        )

        # Store preserved split datasets
        self.train_dataset = Dataset(
            train_df.copy(), target_columns, ignore_columns, metric_columns, name,
            data_id_column, model_id_column, input_column, output_column,
            reference_columns, metrics, task_description=task_description
        )
        self.val_dataset = Dataset(
            train_df.copy(), target_columns, ignore_columns, metric_columns, name,
            data_id_column, model_id_column, input_column, output_column,
            reference_columns, metrics, task_description=task_description
        )
        self.test_dataset = Dataset(
            test_df.copy(), target_columns, ignore_columns, metric_columns, name,
            data_id_column, model_id_column, input_column, output_column,
            reference_columns, metrics, task_description=task_description
        )

        # Initialize parent with training dataframe (consistent with other loaders)
        super().__init__(
            train_df.copy(), target_columns, ignore_columns, metric_columns, name,
            data_id_column, model_id_column, input_column, output_column,
            reference_columns, metrics, task_description=task_description
        )

    def get_splits(self, split_column=None, train_ratio=0.5, val_ratio=0.2, seed=None, max_size=None, *, preserve_splits=True):
        if preserve_splits:
            train_dataset = self.train_dataset
            val_dataset = self.val_dataset
            test_dataset = self.test_dataset

            if max_size:
                # Apply max_size consistently to each split
                train_dataset = train_dataset.get_subset(max_size, seed=42 if seed is None else seed)
                val_dataset = val_dataset.get_subset(max_size, seed=42 if seed is None else seed)
                test_dataset = test_dataset.get_subset(max_size, seed=42 if seed is None else seed)

            return train_dataset, val_dataset, test_dataset
        else:
            return super().get_splits(split_column, train_ratio, val_ratio, seed, max_size=max_size)

    def get_kfold_splits(self, k=5, split_column=None, seed=None, test_ratio=0.3, max_size=None, *, preserve_splits=True):
        if preserve_splits:
            # Keep provided test set fixed; create folds from training data only
            test_dataset = self.test_dataset
            if max_size:
                test_dataset = test_dataset.get_subset(max_size, seed=42 if seed is None else seed)

            splits, train_dataset, _ = self.train_dataset.get_kfold_splits(
                k=k,
                split_column=split_column,
                seed=seed,
                test_ratio=0.0,
                max_size=max_size,
            )
            return splits, train_dataset, test_dataset
        else:
            return super().get_kfold_splits(k, split_column, seed, test_ratio, max_size)

class TauBenchBigger(TauBench):
    def __init__(self, dir_path: str = './autometrics/dataset/datasets/taubench', train_file: str = 'train_runs_full.csv', test_file: str = 'test_runs.csv'):
        super().__init__(dir_path, train_file, test_file, name='taubench_bigger')

class TauBenchHighTemperature(TauBench):
    def __init__(self, dir_path: str = './autometrics/dataset/datasets/taubench', train_file: str = 'train_runs_high_temperature.csv', test_file: str = 'test_runs.csv'):
        super().__init__(dir_path, train_file, test_file, name='taubench_high_temperature')

if __name__ == "__main__":
    TauBench()