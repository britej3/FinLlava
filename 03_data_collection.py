# 03_data_collection.py - Comprehensive Data Collection & Educational Dataset Generation
# Fully Autonomous Algorithmic Crypto High Leveraged Futures Scalping Bot

import os
import sys
import time
import json
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import yfinance as yf
import ccxt
import ta
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from datasets import Dataset, DatasetDict
import yaml
import hashlib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BOT_NAME = "Fully Autonomous Algorithmic Crypto High Leveraged Futures Scalping Bot"
VERSION = "v1.0.0"

class ComprehensiveDataCollector:
    def __init__(self, config_path: str = "/workspace/autonomous_trading_system/config"):
        self.config_path = Path(config_path)
        self.data_path = Path("/workspace/autonomous_trading_system/data")
        self.raw_data_path = self.data_path / "raw"
        self.processed_data_path = self.data_path / "processed"
        
        # Quality control parameters
        self.quality_thresholds = {
            'min_instruction_length': 10,
            'max_instruction_length': 500,
            'min_response_length': 20,
            'max_response_length': 2000,
            'min_similarity_threshold': 0.7,  # For duplicate detection
            'max_repetition_ratio': 0.3,  # Maximum allowed repetitive content
            'required_trading_keywords': [
                'trading', 'price', 'market', 'volume', 'support', 'resistance',
                'bullish', 'bearish', 'trend', 'breakout', 'fibonacci', 'rsi',
                'macd', 'bollinger', 'scalping', 'futures', 'leverage'
            ]
        }
        
        # Dataset statistics tracking
        self.dataset_stats = {
            'total_collected': 0,
            'quality_passed': 0,
            'duplicates_removed': 0,
            'low_quality_filtered': 0,
            'multimodal_examples': 0,
            'text_only_examples': 0
        }
        
        # Load configurations
        self.load_configs()
        
        # Initialize data generation model
        self.init_data_generation_model()
        
        # Initialize exchange connections
        self.init_exchanges()
        
        # Initialize quality evaluator
        self.init_quality_evaluator()
        
    def load_configs(self):
        """Load all configuration files"""
        try:
            with open(self.config_path / "market_config.yaml", 'r') as f:
                self.market_config = yaml.safe_load(f)
            with open(self.config_path / "stage1_config.yaml", 'r') as f:
                self.stage1_config = yaml.safe_load(f)
            logger.info("✅ Configurations loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading configurations: {e}")
            sys.exit(1)
    
    def init_data_generation_model(self):
        """Initialize model for generating instruction-response pairs"""
        try:
            model_name = "mistralai/Mistral-7B-Instruct-v0.2"
            logger.info(f"Loading data generation model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Use 4-bit quantization to save memory
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            logger.info("✅ Data generation model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading data generation model: {e}")
            sys.exit(1)
    
    def init_exchanges(self):
        """Initialize exchange connections"""
        try:
            # Initialize Binance exchange
            self.binance = ccxt.binance({
                'apiKey': self.market_config.get('exchanges', {}).get('binance', {}).get('api_key', ''),
                'secret': self.market_config.get('exchanges', {}).get('binance', {}).get('secret_key', ''),
                'options': {
                    'defaultType': 'future'
                }
            })
            
            # Enable rate limiting
            self.binance.enableRateLimit = True
            
            # Set sandbox mode if specified
            if self.market_config.get('exchanges', {}).get('binance', {}).get('sandbox', False):
                self.binance.set_sandbox_mode(True)
            
            logger.info("✅ Exchange connections initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Exchange initialization failed: {e}")
            self.binance = None
    
    def init_quality_evaluator(self):
        """Initialize quality evaluation components"""
        try:
            # TF-IDF vectorizer for similarity detection
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Quality evaluation model (optional - can use rule-based)
            self.quality_pipeline = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Store processed instruction hashes to detect duplicates
            self.instruction_hashes = set()
            self.instruction_vectors = []
            
            logger.info("✅ Quality evaluator initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Quality evaluator initialization failed: {e}")
            self.quality_pipeline = None
    
    def evaluate_data_quality(self, instruction: str, response: str) -> Tuple[bool, Dict]:
        """
        Comprehensive data quality evaluation for trading instruction-response pairs
        
        Returns:
            Tuple[bool, Dict]: (is_high_quality, quality_metrics)
        """
        quality_metrics = {
            'length_check': False,
            'content_relevance': False,
            'repetition_check': False,
            'trading_relevance': False,
            'uniqueness_check': False,
            'overall_score': 0.0
        }
        
        # 1. Length validation
        inst_len = len(instruction.split())
        resp_len = len(response.split())
        
        if (self.quality_thresholds['min_instruction_length'] <= inst_len <= self.quality_thresholds['max_instruction_length'] and
            self.quality_thresholds['min_response_length'] <= resp_len <= self.quality_thresholds['max_response_length']):
            quality_metrics['length_check'] = True
        
        # 2. Trading relevance check
        trading_keywords_found = sum(1 for keyword in self.quality_thresholds['required_trading_keywords'] 
                                   if keyword.lower() in instruction.lower() + response.lower())
        
        if trading_keywords_found >= 2:  # At least 2 trading-related keywords
            quality_metrics['trading_relevance'] = True
        
        # 3. Repetition check
        response_words = response.lower().split()
        if len(response_words) > 0:
            word_counts = pd.Series(response_words).value_counts()
            repetition_ratio = (word_counts.iloc[0] if len(word_counts) > 0 else 0) / len(response_words)
            if repetition_ratio <= self.quality_thresholds['max_repetition_ratio']:
                quality_metrics['repetition_check'] = True
        
        # 4. Content quality (using sentiment as proxy)
        try:
            if self.quality_pipeline:
                sentiment = self.quality_pipeline(response[:512])  # Limit to 512 chars for speed
                if sentiment[0]['score'] > 0.6:  # High confidence responses
                    quality_metrics['content_relevance'] = True
            else:
                # Fallback rule-based check
                if not any(phrase in response.lower() for phrase in ['i don\'t know', 'not sure', 'unclear']):
                    quality_metrics['content_relevance'] = True
        except:
            quality_metrics['content_relevance'] = True  # Default to true if evaluation fails
        
        # 5. Uniqueness check (avoid duplicates)
        instruction_hash = hashlib.md5(instruction.lower().encode()).hexdigest()
        if instruction_hash not in self.instruction_hashes:
            self.instruction_hashes.add(instruction_hash)
            quality_metrics['uniqueness_check'] = True
            
            # Also check semantic similarity if we have existing vectors
            if len(self.instruction_vectors) > 0:
                try:
                    # Vectorize current instruction
                    current_vector = self.tfidf_vectorizer.transform([instruction])
                    
                    # Check similarity with existing instructions
                    similarities = cosine_similarity(current_vector, self.instruction_vectors).flatten()
                    max_similarity = max(similarities) if len(similarities) > 0 else 0
                    
                    if max_similarity < self.quality_thresholds['min_similarity_threshold']:
                        # Add to our vector collection
                        self.instruction_vectors.append(current_vector)
                    else:
                        quality_metrics['uniqueness_check'] = False
                        
                except Exception as e:
                    logger.debug(f"Similarity check failed: {e}")
        
        # Calculate overall quality score
        quality_score = sum(quality_metrics[key] for key in quality_metrics if key != 'overall_score') / 5.0
        quality_metrics['overall_score'] = quality_score
        
        # High quality threshold: 4/5 checks must pass (0.8)
        is_high_quality = quality_score >= 0.8
        
        return is_high_quality, quality_metrics
    
    def collect_market_data(self):
        """Collect comprehensive market data"""
        logger.info("🔄 Starting market data collection...")
        
        symbols = self.market_config.get('symbols', ['BTCUSDT', 'ETHUSDT'])
        timeframes = self.market_config.get('timeframes', ['1m', '5m', '1h', '1d'])
        
        market_data = {}
        
        for symbol in symbols:
            market_data[symbol] = {}
            logger.info(f"Collecting data for {symbol}")
            
            for timeframe in timeframes:
                try:
                    # Get historical data (90 days)
                    since = int((datetime.now() - timedelta(days=90)).timestamp() * 1000)
                    
                    if hasattr(self, 'binance'):
                        ohlcv = self.binance.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        
                        # Add technical indicators
                        df = self.add_technical_indicators(df)
                        
                        market_data[symbol][timeframe] = df
                        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
       """Add technical indicators to market data"""
       try:
           # Moving averages
           df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
           df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
           df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
           df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
           
           # MACD
           macd = ta.trend.MACD(df['close'])
           df['macd'] = macd.macd()
           df['macd_signal'] = macd.macd_signal()
           df['macd_histogram'] = macd.macd_diff()
           
           # RSI
           df['rsi'] = ta.momentum.rsi(df['close'], window=14)
           
           # Bollinger Bands
           bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
           df['bb_upper'] = bb.bollinger_hband()
           df['bb_middle'] = bb.bollinger_mavg()
           df['bb_lower'] = bb.bollinger_lband()
           
           # Average True Range
           df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
           
           # Stochastic Oscillator
           stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
           df['stoch_k'] = stoch.stoch()
           df['stoch_d'] = stoch.stoch_signal()
           
           # Williams %R
           df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
           
           # Volume indicators
           df['volume_sma'] = ta.volume.volume_sma(df['volume'], window=20)
           
           logger.info("✅ Technical indicators added successfully")
           return df
           
       except Exception as e:
           logger.error(f"❌ Error adding technical indicators: {e}")
           return df
                        
                        # Save to file
                        output_path = self.raw_data_path / "market_data" / f"{symbol}_{timeframe}.csv"
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        df.to_csv(output_path)
                        
                except Exception as e:
                    logger.error(f"❌ Error collecting data for {symbol} {timeframe}: {e}")
                    continue
                        
    def generate_llm_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using the data generation model"""
        try:
            # Format prompt for Mistral
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            # Tokenize
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(formatted_prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating LLM response: {e}")
            return ""
    
    def create_educational_dataset(self):
        """Generate comprehensive educational dataset for trading"""
        logger.info(f"🔄 Creating educational dataset for {BOT_NAME}...")
        
        # Educational content sources
        educational_prompts = {
            "fundamental_concepts": [
                "Explain order book dynamics and how to read Level II data for crypto futures scalping",
                "What are perpetual futures and how does funding rate affect scalping strategies", 
                "Describe cross-exchange arbitrage opportunities in cryptocurrency markets",
                "Explain liquidation cascades and how to identify stop hunting in high leverage trading",
                "How does options flow impact spot and futures prices in crypto markets"
            ],
            "chart_analysis": [
                "How to identify support and resistance levels on different timeframes for scalping",
                "Explain candlestick patterns most relevant for 1-minute crypto futures trading",
                "Describe volume profile analysis for identifying key price levels",
                "What is multi-timeframe confluence and how to use it for entry signals",
                "How to spot market structure breaks in high-frequency trading"
            ],
            "risk_management": [
                "Calculate optimal position size for high leverage crypto futures trading",
                "Explain dynamic stop-loss placement strategies for scalping",
                "How to manage portfolio heat across multiple crypto positions",
                "Describe correlation-based exposure limits in crypto trading",
                "What are the best drawdown recovery strategies for algorithmic trading"
            ],
            "market_microstructure": [
                "How to analyze bid-ask spread patterns in crypto futures",
                "Explain order flow imbalance detection for scalping entries",
                "Identify market maker behavior patterns in cryptocurrency markets",
                "Describe latency arbitrage opportunities in crypto trading",
                "How to spot cross-market inefficiencies between spot and futures"
            ],
            "scalping_strategies": [
                "Best timeframes and indicators for crypto futures scalping",
                "How to use funding rates as a scalping signal",
                "Explain momentum scalping strategies in volatile crypto markets",
                "Describe mean reversion scalping techniques for range-bound markets",
                "How to scalp around key economic events and news releases"
            ],
            "leverage_management": [
                "Optimal leverage levels for different market volatility conditions",
                "How to dynamically adjust leverage based on market regime",
                "Risk management when using 10x+ leverage in crypto futures",
                "Explain the relationship between leverage and position sizing",
                "How to use partial closes to manage high leverage positions"
            ]
        }
        
        all_datasets = []
        total_target = 15000  # Target dataset size
        
        for category, prompts in educational_prompts.items():
            logger.info(f"📚 Generating {category} examples...")
            category_examples = []
            
            # Generate multiple variations per prompt
            for prompt in tqdm(prompts, desc=f"Processing {category}"):
                for variation in range(5):  # 5 variations per prompt
                    try:
                        # Create instruction variation
                        instruction_prompt = f"""
                        As an expert in cryptocurrency futures trading and {BOT_NAME}, create a detailed educational instruction about: {prompt}
                        
                        Make the instruction specific to crypto futures scalping with high leverage. 
                        Include practical examples and be specific about risk management.
                        
                        Instruction:"""
                        
                        instruction = self.generate_llm_response(instruction_prompt, max_tokens=200)
                        
                        if not instruction:
                            continue
                            
                        # Create detailed response
                        response_prompt = f"""
                        As an expert algorithmic trader specializing in {BOT_NAME}, provide a comprehensive answer to this trading question:
                        
                        Question: {instruction}
                        
                        Provide a detailed, practical answer that includes:
                        1. Technical explanation
                        2. Practical implementation steps
                        3. Risk considerations
                        4. Real-world examples from crypto markets
                        5. Common pitfalls to avoid
                        
                        Answer:"""
                        
                        response = self.generate_llm_response(response_prompt, max_tokens=800)
                        
                        if not response:
                            continue
                        
                        # Quality evaluation
                        self.dataset_stats['total_collected'] += 1
                        is_high_quality, quality_metrics = self.evaluate_data_quality(instruction, response)
                        
                        if is_high_quality:
                            example = {
                                "instruction": instruction.strip(),
                                "response": response.strip(),
                                "category": category,
                                "bot_name": BOT_NAME,
                                "quality_score": quality_metrics['overall_score'],
                                "generated_at": datetime.now().isoformat(),
                                "prompt_variation": variation + 1
                            }
                            
                            category_examples.append(example)
                            self.dataset_stats['quality_passed'] += 1
                            self.dataset_stats['text_only_examples'] += 1
                            
                        else:
                            self.dataset_stats['low_quality_filtered'] += 1
                            logger.debug(f"Filtered low quality example: {quality_metrics}")
                        
                        # Rate limiting for API calls
                        time.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(f"❌ Error generating example: {e}")
                        continue
            
            all_datasets.extend(category_examples)
            logger.info(f"✅ Generated {len(category_examples)} high-quality examples for {category}")
        
        # Generate synthetic scenarios
        logger.info("🔄 Generating synthetic market scenarios...")
        synthetic_examples = self.generate_synthetic_scenarios(1000)
        all_datasets.extend(synthetic_examples)
        
        # Final quality check and deduplication
        logger.info("🔄 Final quality check and deduplication...")
        final_dataset = self.final_quality_filter(all_datasets)
        
        # Save dataset
        output_path = self.processed_data_path / "instruction_pairs" / "trading_education_dataset.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(final_dataset, f, indent=2)
        
        # Create HuggingFace dataset
        hf_dataset = Dataset.from_list(final_dataset)
        hf_dataset.save_to_disk(str(self.processed_data_path / "instruction_pairs" / "hf_dataset"))
        
        logger.info(f"✅ Educational dataset created: {len(final_dataset)} examples")
        self.print_dataset_statistics()
        
        return final_dataset

    def generate_synthetic_scenarios(self, num_scenarios: int) -> List[Dict]:
        """Generate synthetic market scenarios for data augmentation"""
        logger.info(f"🔄 Generating {num_scenarios} synthetic market scenarios...")
        synthetic_examples = []

        for i in tqdm(range(num_scenarios), desc="Generating synthetic scenarios"):
            try:
                # 1. Select a random market condition
                condition = np.random.choice([
                    "sudden_volatility_spike",
                    "strong_uptrend",
                    "ranging_market",
                    "liquidity_grab",
                    "news_driven_pump"
                ])

                # 2. Create a detailed instruction based on the condition
                instruction_prompt = f"""
                Analyze the following synthetic market scenario for BTC/USDT and provide a detailed trading plan.

                Scenario: {condition.replace('_', ' ').title()}

                Describe the key characteristics of this scenario, what indicators to watch, and a potential scalping strategy with entry, exit, and stop-loss levels.
                Focus on high-leverage futures trading.
                """

                instruction = self.generate_llm_response(instruction_prompt, max_tokens=200)
                if not instruction:
                    continue

                # 3. Generate a corresponding expert response
                response_prompt = f"""
                As an expert algorithmic trader, provide a detailed response to the following trading instruction:

                Instruction: {instruction}

                Your response should be a comprehensive guide on how to trade this scenario, including risk management specific to high-leverage scalping.
                """

                response = self.generate_llm_response(response_prompt, max_tokens=800)
                if not response:
                    continue

                # 4. Quality evaluation
                self.dataset_stats['total_collected'] += 1
                is_high_quality, quality_metrics = self.evaluate_data_quality(instruction, response)

                if is_high_quality:
                    example = {
                        "instruction": instruction.strip(),
                        "response": response.strip(),
                        "category": "synthetic_scenario",
                        "bot_name": BOT_NAME,
                        "quality_score": quality_metrics['overall_score'],
                        "generated_at": datetime.now().isoformat(),
                        "prompt_variation": 0
                    }
                    synthetic_examples.append(example)
                    self.dataset_stats['quality_passed'] += 1
                    self.dataset_stats['text_only_examples'] += 1
                else:
                    self.dataset_stats['low_quality_filtered'] += 1

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error generating synthetic scenario: {e}")
                continue

        logger.info(f"✅ Generated {len(synthetic_examples)} high-quality synthetic examples")
        return synthetic_examples

    def final_quality_filter(self, dataset: List[Dict]) -> List[Dict]:
        """Perform final quality filtering and deduplication"""

        # Simple deduplication based on instruction
        unique_instructions = set()
        deduplicated_dataset = []

        for example in dataset:
            instruction = example['instruction']
            if instruction not in unique_instructions:
                unique_instructions.add(instruction)
                deduplicated_dataset.append(example)

        self.dataset_stats['duplicates_removed'] = len(dataset) - len(deduplicated_dataset)

        # Further filtering can be added here if needed

        return deduplicated_dataset

    def print_dataset_statistics(self):
        """Print final dataset statistics"""
        logger.info("\n" + "="*50)
        logger.info("📊 Final Dataset Statistics")
        logger.info("="*50)

        for key, value in self.dataset_stats.items():
            logger.info(f"{key.replace('_', ' ').title():<25}: {value}")

        total_generated = self.dataset_stats.get('total_collected', 0)
        if total_generated > 0:
            pass_rate = (self.dataset_stats.get('quality_passed', 0) / total_generated) * 100
            logger.info(f"{'Quality Pass Rate':<25}: {pass_rate:.2f}%")

        logger.info("="*50 + "\n")