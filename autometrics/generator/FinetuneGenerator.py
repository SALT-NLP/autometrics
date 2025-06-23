import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List
from sklearn.model_selection import train_test_split
import dspy
import torch
from platformdirs import user_data_dir

from autometrics.generator.Generator import Generator
from autometrics.util.format import get_default_formatter

# Import the metric classes (will be created next)
from autometrics.metrics.generated.GeneratedFinetunedMetric import (
    GeneratedRefFreeFinetunedMetric,
    GeneratedRefBasedFinetunedMetric
)


class FinetuneGenerator(Generator):
    """Generate fine-tuned metrics by training ModernBERT-Large models on user data.
    
    This generator fine-tunes a regression model on the provided dataset, creating
    metrics that can predict quality scores based on the learned patterns in the data.
    Unlike other generators, this creates actual learned models rather than prompting strategies.
    
    The class follows the Generator interface but has special considerations:
    - Default n_metrics=1 (fine-tuning is expensive)
    - Models are saved to user data directory
    - Supports optional HuggingFace upload
    - Uses 80/20 train/validation split
    """

    def __init__(
        self,
        name: str = "FinetuneGenerator",
        description: str = "Generate fine-tuned ModernBERT metrics based on dataset regression training",
        generator_llm: Optional[dspy.LM] = None,
        executor_class: type | None = None,
        executor_kwargs: dict | None = None,
        model_name: str = "answerdotai/ModernBERT-large",
        max_seq_length: int = 2048,
        num_train_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        upload_to_hf: bool = False,
        hf_repo_name: Optional[str] = None,
        seed: int = 42,
    ):
        super().__init__(
            name=name,
            description=description,
            generator_llm=generator_llm,
            executor_class=executor_class,
            executor_kwargs=executor_kwargs or {},
        )

        # Fine-tuning specific parameters
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.upload_to_hf = upload_to_hf
        self.hf_repo_name = hf_repo_name
        self.seed = seed

        # Guarantee attribute is a dictionary for ** expansion later
        if self.executor_kwargs is None:
            self.executor_kwargs = {}

        # Set up model save directory
        self.model_save_dir = Path(user_data_dir("autometrics")) / "models"
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

    def _get_formatter(self, dataset):
        """Get the appropriate formatter for the dataset."""
        if not dataset:
            return lambda x: str(x)
        return get_default_formatter(dataset)

    def _determine_executor_class(self, dataset):
        """Determine whether to use reference-based or reference-free metrics based on dataset."""
        reference_columns = dataset.get_reference_columns()
        has_references = reference_columns is not None and len(reference_columns) > 0
        
        if has_references:
            return GeneratedRefBasedFinetunedMetric
        else:
            return GeneratedRefFreeFinetunedMetric

    def _prepare_training_data(self, dataset, target_measure: str, formatter: Optional[Callable] = None):
        """Prepare the dataset for training by splitting and formatting."""
        if not formatter:
            formatter = self._get_formatter(dataset)

        df = dataset.get_dataframe()
        if not target_measure:
            target_measure = dataset.get_target_columns()[0]

        # Extract input, output, target, and references
        input_col = dataset.get_input_column()
        output_col = dataset.get_output_column()
        reference_cols = dataset.get_reference_columns()
        
        # Create text for training using the dataset's formatter
        texts = []
        for _, row in df.iterrows():
            formatted_text = formatter(row)
            texts.append(formatted_text)

        # Get target values
        targets = df[target_measure].values

        # 80/20 train/validation split
        train_texts, val_texts, train_targets, val_targets = train_test_split(
            texts, targets, test_size=0.2, random_state=self.seed, stratify=None
        )

        return train_texts, val_texts, train_targets, val_targets

    def _finetune_model(self, train_texts: List[str], train_targets: np.ndarray, 
                       val_texts: List[str], val_targets: np.ndarray, 
                       model_save_path: str) -> str:
        """Fine-tune the ModernBERT model for regression."""
        try:
            from unsloth import FastModel
            from transformers import (
                AutoModelForSequenceClassification, 
                TrainingArguments, 
                Trainer,
                training_args
            )
            from datasets import Dataset
            import torch.nn.functional as F
            from sklearn.metrics import mean_squared_error
        except ImportError as e:
            raise ImportError(f"Required libraries not installed: {e}. Please install unsloth and transformers.")

        print(f"Fine-tuning {self.model_name} for regression...")
        
        # Load model with unsloth
        model, tokenizer = FastModel.from_pretrained(
            model_name=self.model_name,
            load_in_4bit=False,
            max_seq_length=self.max_seq_length,
            dtype=None,
            auto_model=AutoModelForSequenceClassification,
            num_labels=1,  # Regression - single output
        )

        # Make all parameters trainable
        for param in model.parameters():
            param.requires_grad = True

        # Prepare datasets
        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, padding=True, max_length=self.max_seq_length)

        train_dataset = Dataset.from_dict({
            'text': train_texts,
            'labels': train_targets.astype(float)
        })
        val_dataset = Dataset.from_dict({
            'text': val_texts, 
            'labels': val_targets.astype(float)
        })

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)

        # Define compute metrics for regression
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            # For regression, predictions is logits with shape (batch_size, 1)
            predictions = predictions.flatten()
            mse = mean_squared_error(labels, predictions)
            return {"mse": mse, "rmse": np.sqrt(mse)}

        # Set up training arguments
        training_args_config = TrainingArguments(
            output_dir=model_save_path,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=1,
            warmup_steps=10,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim=training_args.OptimizerNames.ADAMW_TORCH,
            learning_rate=self.learning_rate,
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            seed=self.seed,  # Use consistent seed
            num_train_epochs=self.num_train_epochs,
            save_strategy="epoch",
            report_to="none",
            group_by_length=True,
            eval_strategy="steps",
            eval_steps=0.25,
            logging_strategy="steps",
            logging_steps=0.25,
            load_best_model_at_end=True,
            metric_for_best_model="mse",
            greater_is_better=False,  # Lower MSE is better
            # Early stopping based on validation performance
            save_total_limit=3,  # Keep only best 3 checkpoints
            early_stopping_patience=5,  # Stop if no improvement for 5 eval steps
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args_config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        # Train the model
        print("Starting training...")
        trainer_stats = trainer.train()
        
        # Save the final model
        trainer.save_model()
        tokenizer.save_pretrained(model_save_path)
        
        print(f"Model saved to {model_save_path}")
        print(f"Training completed. Final train loss: {trainer_stats.training_history[-1].get('train_loss', 'N/A')}")

        # Upload to HuggingFace if requested
        if self.upload_to_hf and self.hf_repo_name:
            try:
                trainer.push_to_hub(self.hf_repo_name)
                print(f"Model uploaded to HuggingFace: {self.hf_repo_name}")
            except Exception as e:
                print(f"Failed to upload to HuggingFace: {e}")

        return model_save_path

    def generate(self, dataset, target_measure: Optional[str] = None, n_metrics: int = 1, 
                formatter: Optional[Callable] = None, **kwargs) -> List:
        """
        Generate fine-tuned metrics based on the dataset.
        
        Note that n_metrics defaults to 1 for fine-tuning since it's computationally expensive.
        """
        task_description = dataset.get_task_description()

        if not formatter:
            formatter = self._get_formatter(dataset)
        
        # Step-1: Determine the appropriate executor class based on dataset
        if self.executor_class is None:
            dynamic_executor_class = self._determine_executor_class(dataset)
        else:
            dynamic_executor_class = self.executor_class

        # Step-2: Prepare training data
        if not target_measure:
            target_measure = dataset.get_target_columns()[0]
            
        print(f"Preparing training data for target measure: {target_measure}")
        train_texts, val_texts, train_targets, val_targets = self._prepare_training_data(
            dataset, target_measure, formatter
        )

        print(f"Training set size: {len(train_texts)}")
        print(f"Validation set size: {len(val_texts)}")

        # Step-3: Generate metrics (typically just 1 for fine-tuning)
        new_metrics = []
        
        for i in range(n_metrics):
            # Create unique model name with seed for reproducibility
            safe_dataset_name = dataset.get_name().replace(" ", "_").replace("/", "_")
            safe_target_name = target_measure.replace(" ", "_").replace("/", "_")
            model_name = f"finetuned_{safe_dataset_name}_{safe_target_name}_seed{self.seed}_{i+1}"
            
            model_save_path = self.model_save_dir / model_name
            model_save_path.mkdir(exist_ok=True)
            
            # Step-4: Fine-tune the model
            print(f"Fine-tuning model {i+1}/{n_metrics}: {model_name}")
            final_model_path = self._finetune_model(
                train_texts, train_targets, 
                val_texts, val_targets,
                str(model_save_path)
            )

            # Step-5: Create the metric instance
            # Note: Fine-tuned metrics don't need an LLM for metric card generation
            # They generate cards programmatically using template-based approach
            metric = dynamic_executor_class(
                name=f"{model_name}_ModernBERT",
                description=f"Fine-tuned ModernBERT metric for {target_measure} on {dataset.get_name()}",
                model_path=final_model_path,
                task_description=task_description,
                target_measure=target_measure,
                dataset_name=dataset.get_name(),
                training_stats={
                    "train_size": len(train_texts),
                    "val_size": len(val_texts),
                    "target_mean": float(np.mean(train_targets)),
                    "target_std": float(np.std(train_targets)),
                    "epochs": self.num_train_epochs,
                    "learning_rate": self.learning_rate,
                },
                metric_card_author_model=None,  # No LLM needed for programmatic generation
                **self.executor_kwargs,
            )

            new_metrics.append(metric)

        return new_metrics

    def get_name(self):
        return self.name
    
    def get_description(self):
        return self.description

    def __str__(self):
        return f"{self.name}: {self.description}"
    
    def __repr__(self):
        return self.__str__() 