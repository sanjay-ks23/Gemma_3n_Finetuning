"""
Data preprocessing module for therapeutic conversation datasets.
Handles CounselChat and AnnoMI dataset preprocessing, formatting, and preparation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datasets import Dataset, DatasetDict, load_dataset
import re
from pathlib import Path
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

from utils import Logger


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Conversation:
    """Represents a complete conversation."""
    turns: List[ConversationTurn]
    conversation_id: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_chat_format(self) -> List[Dict[str, str]]:
        """Convert to standard chat format."""
        return [{"role": turn.role, "content": turn.content} for turn in self.turns]
    
    def to_prompt_completion_format(self) -> Tuple[str, str]:
        """Convert to prompt-completion format for SFT."""
        if len(self.turns) < 2:
            raise ValueError("Conversation must have at least 2 turns")
        
        # Combine all user messages as prompt
        user_messages = [turn.content for turn in self.turns if turn.role == "user"]
        assistant_messages = [turn.content for turn in self.turns if turn.role == "assistant"]
        
        if not user_messages or not assistant_messages:
            raise ValueError("Conversation must have both user and assistant messages")
        
        prompt = " ".join(user_messages)
        completion = " ".join(assistant_messages)
        
        return prompt, completion


class BaseDatasetProcessor(ABC):
    """Abstract base class for dataset processors."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or Logger("DataProcessor").get_logger()
    
    @abstractmethod
    def load_raw_data(self) -> Any:
        """Load raw dataset."""
        pass
    
    @abstractmethod
    def process_conversations(self, raw_data: Any) -> List[Conversation]:
        """Process raw data into conversation format."""
        pass
    
    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information and statistics."""
        pass
    
    def validate_conversation(self, conversation: Conversation) -> bool:
        """Validate conversation quality with more lenient rules."""
        # Basic validation rules
        if len(conversation.turns) < 2:
            return False
        
        # Check for minimum content length (reduced from 10 to 3 characters)
        for turn in conversation.turns:
            if len(turn.content.strip()) < 3:
                return False
        
        # More flexible role pattern validation
        roles = [turn.role for turn in conversation.turns]
        if not self._has_valid_role_pattern_flexible(roles):
            return False
        
        return True
    
    def _has_valid_role_pattern_flexible(self, roles: List[str]) -> bool:
        """More flexible role pattern validation."""
        # Must have both user and assistant roles
        if 'user' not in roles or 'assistant' not in roles:
            return False
        
        # Allow up to 3 consecutive same roles (more realistic for therapy conversations)
        consecutive_count = 1
        for i in range(1, len(roles)):
            if roles[i] == roles[i-1]:
                consecutive_count += 1
                if consecutive_count > 3:  # Allow up to 3 consecutive
                    return False
            else:
                consecutive_count = 1
        
        return True
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters that might interfere with training
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        # Normalize quotes - use string replacement instead of regex
        text = text.replace('"', '"').replace('"', '"')  # Smart double quotes to regular double quotes
        text = text.replace(''', "'").replace(''', "'")  # Smart single quotes to regular single quotes
        
        return text


class CounselChatProcessor(BaseDatasetProcessor):
    """Processor for CounselChat dataset from HuggingFace."""
    
    def __init__(self, dataset_name: str = "nbertagnolli/counsel-chat", **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.raw_data = None
    
    def load_raw_data(self) -> Dataset:
        """Load CounselChat dataset from HuggingFace."""
        self.logger.info(f"Loading CounselChat dataset: {self.dataset_name}")
        
        try:
            dataset = load_dataset(self.dataset_name)
            self.raw_data = dataset['train'] if 'train' in dataset else dataset
            self.logger.info(f"Loaded {len(self.raw_data)} samples from CounselChat")
            return self.raw_data
        except Exception as e:
            self.logger.error(f"Failed to load CounselChat dataset: {e}")
            raise
    
    def process_conversations(self, raw_data: Optional[Dataset] = None) -> List[Conversation]:
        """Process CounselChat data into conversation format."""
        if raw_data is None:
            raw_data = self.raw_data or self.load_raw_data()
        
        conversations = []
        rejected_reasons = {'too_short_content': 0, 'invalid_roles': 0, 'too_few_turns': 0, 'empty_content': 0}
        
        for idx, sample in enumerate(raw_data):
            try:
                # CounselChat format: questionTitle, questionText, answerText
                question_title = sample.get('questionTitle', '')
                question_text = sample.get('questionText', '')
                answer_text = sample.get('answerText', '')
                
                # Combine question title and text
                user_content = f"{question_title}\n{question_text}".strip()
                user_content = self.clean_text(user_content)
                
                assistant_content = self.clean_text(answer_text)
                
                if not user_content or not assistant_content:
                    rejected_reasons['empty_content'] += 1
                    continue
                
                # Create conversation
                turns = [
                    ConversationTurn(role="user", content=user_content),
                    ConversationTurn(role="assistant", content=assistant_content)
                ]
                
                conversation = Conversation(
                    turns=turns,
                    conversation_id=f"counselchat_{idx}",
                    metadata={
                        'source': 'counselchat',
                        'topic': sample.get('topic', 'general'),
                        'original_index': idx
                    }
                )
                
                if self.validate_conversation(conversation):
                    conversations.append(conversation)
                else:
                    # Debug why it was rejected
                    if len(conversation.turns) < 2:
                        rejected_reasons['too_few_turns'] += 1
                    elif any(len(turn.content.strip()) < 3 for turn in conversation.turns):
                        rejected_reasons['too_short_content'] += 1
                    else:
                        rejected_reasons['invalid_roles'] += 1
                    
            except Exception as e:
                self.logger.warning(f"Error processing CounselChat sample {idx}: {e}")
                continue
        
        self.logger.info(f"Processed {len(conversations)} valid conversations from CounselChat")
        self.logger.info(f"Rejection reasons: {rejected_reasons}")
        return conversations
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get CounselChat dataset information."""
        if self.raw_data is None:
            self.load_raw_data()
        
        return {
            'name': 'CounselChat',
            'source': self.dataset_name,
            'total_samples': len(self.raw_data),
            'columns': list(self.raw_data.column_names),
            'description': 'Mental health counseling conversations'
        }


class AnnoMIProcessor(BaseDatasetProcessor):
    """Processor for AnnoMI dataset (CSV format)."""
    
    def __init__(self, csv_path: str, **kwargs):
        super().__init__(**kwargs)
        self.csv_path = Path(csv_path)
        self.raw_data = None
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load AnnoMI dataset from CSV."""
        self.logger.info(f"Loading AnnoMI dataset from: {self.csv_path}")
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"AnnoMI CSV file not found: {self.csv_path}")
        
        try:
            self.raw_data = pd.read_csv(self.csv_path)
            self.logger.info(f"Loaded {len(self.raw_data)} rows from AnnoMI dataset")
            self.logger.info(f"Columns: {list(self.raw_data.columns)}")
            return self.raw_data
        except Exception as e:
            self.logger.error(f"Failed to load AnnoMI dataset: {e}")
            raise
    
    def process_conversations(self, raw_data: Optional[pd.DataFrame] = None) -> List[Conversation]:
        """Process AnnoMI data into conversation format."""
        if raw_data is None:
            raw_data = self.raw_data or self.load_raw_data()
        
        # Group by transcript_id to create conversations
        conversations = []
        rejected_reasons = {'too_short_content': 0, 'invalid_roles': 0, 'too_few_turns': 0, 'no_valid_turns': 0}
        grouped = raw_data.groupby('transcript_id')
        
        self.logger.info(f"Processing {len(grouped)} unique transcripts from AnnoMI")
        
        for transcript_id, group in grouped:
            try:
                # Sort by utterance_id to maintain conversation order
                group = group.sort_values('utterance_id')
                
                turns = []
                for _, row in group.iterrows():
                    # Determine role based on interlocutor
                    role = "assistant" if row['interlocutor'] == 'therapist' else "user"
                    content = self.clean_text(str(row['utterance_text']))
                    
                    if content:
                        turn = ConversationTurn(
                            role=role,
                            content=content,
                            metadata={
                                'timestamp': row.get('timestamp'),
                                'main_therapist_behaviour': row.get('main_therapist_behaviour'),
                                'client_talk_type': row.get('client_talk_type'),
                                'mi_quality': row.get('mi_quality')
                            }
                        )
                        turns.append(turn)
                
                if len(turns) >= 2:
                    conversation = Conversation(
                        turns=turns,
                        conversation_id=f"annomi_{transcript_id}",
                        metadata={
                            'source': 'annomi',
                            'transcript_id': transcript_id,
                            'topic': group.iloc[0].get('topic', 'motivational_interviewing'),
                            'video_title': group.iloc[0].get('video_title'),
                            'mi_quality': group.iloc[0].get('mi_quality')
                        }
                    )
                    
                    if self.validate_conversation(conversation):
                        conversations.append(conversation)
                    else:
                        # Debug why it was rejected
                        if len(conversation.turns) < 2:
                            rejected_reasons['too_few_turns'] += 1
                        elif any(len(turn.content.strip()) < 3 for turn in conversation.turns):
                            rejected_reasons['too_short_content'] += 1
                        else:
                            rejected_reasons['invalid_roles'] += 1
                else:
                    rejected_reasons['no_valid_turns'] += 1
                        
            except Exception as e:
                self.logger.warning(f"Error processing AnnoMI transcript {transcript_id}: {e}")
                continue
        
        self.logger.info(f"Processed {len(conversations)} valid conversations from AnnoMI")
        self.logger.info(f"Rejection reasons: {rejected_reasons}")
        return conversations
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get AnnoMI dataset information."""
        if self.raw_data is None:
            self.load_raw_data()
        
        return {
            'name': 'AnnoMI',
            'source': str(self.csv_path),
            'total_rows': len(self.raw_data),
            'unique_transcripts': self.raw_data['transcript_id'].nunique(),
            'columns': list(self.raw_data.columns),
            'mi_quality_distribution': self.raw_data['mi_quality'].value_counts().to_dict(),
            'description': 'Annotated Motivational Interviewing conversations'
        }


class DatasetFormatter:
    """Formats processed conversations for different training paradigms."""
    
    def __init__(self, tokenizer, max_length: int = 2048, logger: Optional[logging.Logger] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.logger = logger or Logger("DatasetFormatter").get_logger()
    
    def format_for_sft(self, conversations: List[Conversation]) -> Dataset:
        """Format conversations for Supervised Fine-Tuning."""
        formatted_data = []
        
        for conversation in conversations:
            try:
                # Convert to chat format
                messages = conversation.to_chat_format()
                
                # Apply chat template
                formatted_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                # Tokenize and check length
                tokens = self.tokenizer(
                    formatted_text,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors=None
                )
                
                if len(tokens['input_ids']) < 50:  # Skip very short conversations
                    continue
                
                formatted_data.append({
                    'text': formatted_text,
                    'conversation_id': conversation.conversation_id,
                    'source': conversation.metadata.get('source', 'unknown'),
                    'length': len(tokens['input_ids'])
                })
                
            except Exception as e:
                self.logger.warning(f"Error formatting conversation {conversation.conversation_id}: {e}")
                continue
        
        self.logger.info(f"Formatted {len(formatted_data)} conversations for SFT")
        return Dataset.from_list(formatted_data)
    
    def format_for_dpo(self, conversations: List[Conversation], 
                      preference_pairs: Optional[List[Dict]] = None) -> Dataset:
        """Format conversations for Direct Preference Optimization."""
        if preference_pairs is None:
            # Generate synthetic preference pairs from conversations
            preference_pairs = self._generate_preference_pairs(conversations)
        
        formatted_data = []
        
        for pair in preference_pairs:
            try:
                prompt = pair['prompt']
                chosen = pair['chosen']
                rejected = pair['rejected']
                
                # Tokenize to check lengths
                prompt_tokens = self.tokenizer(prompt, return_tensors=None)
                chosen_tokens = self.tokenizer(chosen, return_tensors=None)
                rejected_tokens = self.tokenizer(rejected, return_tensors=None)
                
                # Skip if too long
                total_length = len(prompt_tokens['input_ids']) + max(
                    len(chosen_tokens['input_ids']), 
                    len(rejected_tokens['input_ids'])
                )
                
                if total_length > self.max_length:
                    continue
                
                formatted_data.append({
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected,
                    'source': pair.get('source', 'synthetic')
                })
                
            except Exception as e:
                self.logger.warning(f"Error formatting DPO pair: {e}")
                continue
        
        self.logger.info(f"Formatted {len(formatted_data)} preference pairs for DPO")
        return Dataset.from_list(formatted_data)
    
    def _generate_preference_pairs(self, conversations: List[Conversation]) -> List[Dict]:
        """Generate synthetic preference pairs from conversations."""
        # This is a simplified approach - in practice, you'd want more sophisticated methods
        pairs = []
        
        for conversation in conversations:
            if len(conversation.turns) >= 4:  # Need at least 2 exchanges
                try:
                    # Use first user message as prompt
                    prompt = conversation.turns[0].content
                    
                    # Use first assistant response as chosen
                    chosen = conversation.turns[1].content
                    
                    # Generate a "rejected" response (simplified approach)
                    # In practice, you'd use a weaker model or apply degradation
                    rejected = self._generate_rejected_response(chosen)
                    
                    pairs.append({
                        'prompt': prompt,
                        'chosen': chosen,
                        'rejected': rejected,
                        'source': conversation.metadata.get('source', 'unknown'),
                        'conversation_id': conversation.conversation_id
                    })
                    
                except Exception as e:
                    self.logger.debug(f"Error generating preference pair: {e}")
                    continue
        
        return pairs
    
    def _generate_rejected_response(self, chosen_response: str) -> str:
        """Generate a rejected response (simplified approach)."""
        # This is a placeholder - in practice, you'd use more sophisticated methods
        # such as using a weaker model, adding noise, or using actual negative examples
        
        # Simple approach: make response shorter and less helpful
        sentences = chosen_response.split('.')
        if len(sentences) > 2:
            return '. '.join(sentences[:len(sentences)//2]) + '.'
        else:
            return "I'm not sure how to help with that."


class DatasetManager:
    """Manages multiple datasets and provides unified interface."""
    
    def __init__(self, tokenizer, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger or Logger("DatasetManager").get_logger()
        self.processors = {}
        self.formatter = DatasetFormatter(tokenizer, config.get('max_length', 2048), logger)
    
    def add_processor(self, name: str, processor: BaseDatasetProcessor):
        """Add a dataset processor."""
        self.processors[name] = processor
        self.logger.info(f"Added processor for dataset: {name}")
    
    def process_all_datasets(self) -> Dict[str, List[Conversation]]:
        """Process all registered datasets."""
        all_conversations = {}
        
        for name, processor in self.processors.items():
            self.logger.info(f"Processing dataset: {name}")
            try:
                conversations = processor.process_conversations()
                all_conversations[name] = conversations
                
                # Log dataset info
                info = processor.get_dataset_info()
                self.logger.info(f"Dataset {name} info: {info}")
                
            except Exception as e:
                self.logger.error(f"Failed to process dataset {name}: {e}")
                all_conversations[name] = []
        
        return all_conversations
    
    def create_training_datasets(self, conversations_dict: Dict[str, List[Conversation]], 
                               stage: str = "sft") -> DatasetDict:
        """Create training datasets for specified stage."""
        if stage == "sft":
            return self._create_sft_datasets(conversations_dict)
        elif stage == "dpo":
            return self._create_dpo_datasets(conversations_dict)
        else:
            raise ValueError(f"Unknown training stage: {stage}")
    
    def _create_sft_datasets(self, conversations_dict: Dict[str, List[Conversation]]) -> DatasetDict:
        """Create SFT training datasets."""
        all_conversations = []
        for dataset_name, conversations in conversations_dict.items():
            all_conversations.extend(conversations)
        
        # Shuffle conversations
        np.random.shuffle(all_conversations)
        
        # Split into train/validation
        split_idx = int(len(all_conversations) * 0.9)
        train_conversations = all_conversations[:split_idx]
        val_conversations = all_conversations[split_idx:]
        
        # Format datasets
        train_dataset = self.formatter.format_for_sft(train_conversations)
        val_dataset = self.formatter.format_for_sft(val_conversations)
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    
    def _create_dpo_datasets(self, conversations_dict: Dict[str, List[Conversation]]) -> DatasetDict:
        """Create DPO training datasets."""
        all_conversations = []
        for dataset_name, conversations in conversations_dict.items():
            all_conversations.extend(conversations)
        
        # Create preference pairs
        train_dataset = self.formatter.format_for_dpo(all_conversations)
        
        # Split for validation
        split_idx = int(len(train_dataset) * 0.9)
        val_dataset = Dataset.from_dict(train_dataset[split_idx:])
        train_dataset = Dataset.from_dict(train_dataset[:split_idx])
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    
    def get_dataset_statistics(self, conversations_dict: Dict[str, List[Conversation]]) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        stats = {}
        
        for dataset_name, conversations in conversations_dict.items():
            if not conversations:
                continue
            
            # Basic stats
            total_conversations = len(conversations)
            total_turns = sum(len(conv.turns) for conv in conversations)
            
            # Length statistics
            conversation_lengths = [len(conv.turns) for conv in conversations]
            turn_lengths = []
            for conv in conversations:
                for turn in conv.turns:
                    turn_lengths.append(len(turn.content.split()))
            
            stats[dataset_name] = {
                'total_conversations': total_conversations,
                'total_turns': total_turns,
                'avg_turns_per_conversation': np.mean(conversation_lengths),
                'avg_words_per_turn': np.mean(turn_lengths) if turn_lengths else 0,
                'min_turns': min(conversation_lengths) if conversation_lengths else 0,
                'max_turns': max(conversation_lengths) if conversation_lengths else 0,
                'sources': list(set(conv.metadata.get('source', 'unknown') for conv in conversations))
            }
        
        return stats
