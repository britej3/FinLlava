import torch
from transformers import (
    TrainingArguments,
    Trainer,
    LlavaForConditionalGeneration,
    LlavaProcessor
)
from datasets import load_from_disk
from pathlib import Path
import yaml
import logging
import sys

# Add project root to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from Scripts.FinLlava.core_architecture import create_trading_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BOT_NAME = "Fully Autonomous Algorithmic Crypto High Leveraged Futures Scalping Bot"
VERSION = "v1.0.0"

class Stage1Trainer:
    def __init__(self, config_path: str = "config/stage1_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.base_path = Path("/workspace/autonomous_trading_system")
        self.model_output_dir = self.base_path / "data" / "models" / "stage1_finllava"
        self.model_output_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self) -> dict:
        """Load Stage 1 configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("✅ Stage 1 configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"❌ Error loading Stage 1 configuration: {e}")
            sys.exit(1)

    def load_dataset(self):
        """Load the processed educational dataset"""
        dataset_path = self.base_path / "data" / "processed" / "instruction_pairs" / "hf_dataset"
        try:
            dataset = load_from_disk(str(dataset_path))
            logger.info(f"✅ Dataset loaded from {dataset_path}")

            # Split into train and validation
            if 'train' not in dataset or 'test' not in dataset:
                dataset = dataset.train_test_split(test_size=self.config['data']['validation_split'])

            return dataset['train'], dataset['test']
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            sys.exit(1)

    def preprocess_data(self, examples, processor):
        """Preprocess data for FinLLaVA model"""
        # This function needs to be adapted based on the actual dataset structure
        # Assuming 'instruction' is the text and no image is present for now
        texts = [f"Instruction: {instruction}\nResponse:" for instruction in examples['instruction']]

        inputs = processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config['training']['max_seq_length']
        )

        # Set labels for language modeling
        inputs['labels'] = inputs['input_ids'].clone()

        return inputs

    def train(self):
        """Run the Stage 1 training pipeline"""
        logger.info(f"🚀 Starting Stage 1 Training for {BOT_NAME}...")

        # 1. Create model and processor
        model = create_trading_model("stage1")
        processor = model.processor # The processor is part of the model in our architecture

        # 2. Load and preprocess dataset
        train_dataset, eval_dataset = self.load_dataset()

        # Apply preprocessing
        train_dataset = train_dataset.map(lambda x: self.preprocess_data(x, processor), batched=True)
        eval_dataset = eval_dataset.map(lambda x: self.preprocess_data(x, processor), batched=True)

        # 3. Define Training Arguments
        training_args = TrainingArguments(
            output_dir=str(self.model_output_dir),
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=float(self.config['training']['learning_rate']),
            warmup_steps=self.config['training']['warmup_steps'],
            logging_dir=self.base_path / "logs" / "stage1_training",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,  # Enable mixed-precision training
            report_to="tensorboard",
        )

        # 4. Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # 5. Start Training
        logger.info("🏋️‍♂️ Starting model training...")
        try:
            trainer.train()
            logger.info("✅ Training completed successfully!")
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise

        # 6. Save final model
        logger.info("💾 Saving final model and processor...")
        trainer.save_model(str(self.model_output_dir / "final"))
        processor.save_pretrained(str(self.model_output_dir / "final"))

        logger.info(f"✅ Stage 1 training finished. Model saved to {self.model_output_dir / 'final'}")

if __name__ == "__main__":
    trainer = Stage1Trainer()
    trainer.train()
    logger.info("📝 Next: Run 06_drl_environment.py")
