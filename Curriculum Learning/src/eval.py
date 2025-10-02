"""
Evaluation module for therapeutic conversation models.
Provides comprehensive evaluation metrics and analysis tools.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from pathlib import Path
import json
import pandas as pd
from collections import defaultdict
import re
from dataclasses import dataclass
import time

# Evaluation metrics
from evaluate import load
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize, sent_tokenize

from utils import Logger, ModelUtils


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metric_name: str
    score: float
    details: Optional[Dict[str, Any]] = None


@dataclass
class ConversationEvaluation:
    """Container for conversation-level evaluation."""
    conversation_id: str
    metrics: List[EvaluationResult]
    generated_response: str
    reference_response: str
    prompt: str
    metadata: Optional[Dict[str, Any]] = None


class TherapeuticMetrics:
    """Specialized metrics for therapeutic conversation evaluation."""
    
    def __init__(self):
        # Therapeutic quality indicators
        self.empathy_keywords = [
            'understand', 'feel', 'difficult', 'challenging', 'support', 'here for you',
            'listen', 'care', 'important', 'valid', 'acknowledge', 'recognize'
        ]
        
        self.reflection_patterns = [
            r"it sounds like",
            r"what I hear is",
            r"so you're saying",
            r"if I understand correctly",
            r"let me reflect back",
            r"what I'm hearing"
        ]
        
        self.question_patterns = [
            r"\?",  # Direct questions
            r"can you tell me",
            r"how do you feel",
            r"what do you think",
            r"would you like to"
        ]
        
        self.mi_techniques = {
            'open_questions': [
                r"how", r"what", r"when", r"where", r"why", r"tell me about",
                r"describe", r"explain", r"help me understand"
            ],
            'reflections': [
                r"it sounds like", r"so you", r"what I hear", r"you feel",
                r"you're saying", r"it seems"
            ],
            'affirmations': [
                r"that's great", r"good job", r"you're right", r"that makes sense",
                r"I appreciate", r"that's important", r"you've done well"
            ],
            'summaries': [
                r"so far we've discussed", r"to summarize", r"let me recap",
                r"what I've heard", r"the main points"
            ]
        }
    
    def calculate_empathy_score(self, text: str) -> float:
        """Calculate empathy score based on keyword presence."""
        text_lower = text.lower()
        empathy_count = sum(1 for keyword in self.empathy_keywords if keyword in text_lower)
        return min(empathy_count / len(self.empathy_keywords), 1.0)
    
    def calculate_reflection_score(self, text: str) -> float:
        """Calculate reflection score based on pattern matching."""
        reflection_count = sum(1 for pattern in self.reflection_patterns 
                             if re.search(pattern, text, re.IGNORECASE))
        return min(reflection_count / 3, 1.0)  # Normalize to max 3 reflections
    
    def calculate_question_ratio(self, text: str) -> float:
        """Calculate ratio of questions to total sentences."""
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
        
        question_count = sum(1 for pattern in self.question_patterns 
                           if re.search(pattern, text, re.IGNORECASE))
        return question_count / len(sentences)
    
    def calculate_mi_techniques_score(self, text: str) -> Dict[str, float]:
        """Calculate Motivational Interviewing techniques usage."""
        scores = {}
        text_lower = text.lower()
        
        for technique, patterns in self.mi_techniques.items():
            count = sum(1 for pattern in patterns if re.search(pattern, text_lower))
            scores[technique] = min(count / 2, 1.0)  # Normalize to max 2 per technique
        
        return scores
    
    def calculate_response_length_appropriateness(self, text: str, 
                                                target_min: int = 20, 
                                                target_max: int = 200) -> float:
        """Calculate appropriateness of response length."""
        word_count = len(word_tokenize(text))
        
        if target_min <= word_count <= target_max:
            return 1.0
        elif word_count < target_min:
            return word_count / target_min
        else:
            # Penalize overly long responses
            return max(0.5, target_max / word_count)
    
    def calculate_therapeutic_quality_score(self, text: str) -> Dict[str, float]:
        """Calculate comprehensive therapeutic quality score."""
        return {
            'empathy': self.calculate_empathy_score(text),
            'reflection': self.calculate_reflection_score(text),
            'question_ratio': self.calculate_question_ratio(text),
            'length_appropriateness': self.calculate_response_length_appropriateness(text),
            **self.calculate_mi_techniques_score(text)
        }


class ModelEvaluator:
    """Comprehensive model evaluator for therapeutic conversations."""
    
    def __init__(self, 
                 model_path: str,
                 tokenizer_path: Optional[str] = None,
                 device: str = "auto",
                 logger: Optional[logging.Logger] = None):
        
        self.model_path = Path(model_path)
        self.tokenizer_path = tokenizer_path or model_path
        self.device = device
        self.logger = logger or Logger("ModelEvaluator").get_logger()
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.therapeutic_metrics = TherapeuticMetrics()
        
        # Load evaluation metrics
        self._load_evaluation_metrics()
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
    
    def _load_evaluation_metrics(self):
        """Load standard evaluation metrics."""
        try:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.bleu_smoother = SmoothingFunction().method1
            self.logger.info("Loaded standard evaluation metrics")
        except Exception as e:
            self.logger.warning(f"Failed to load some evaluation metrics: {e}")
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        self.logger.info(f"Loading model from: {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True
            )
            
            self.model.eval()
            self.logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_response(self, 
                         prompt: str, 
                         max_new_tokens: int = 256,
                         temperature: float = 0.7,
                         top_p: float = 0.9,
                         do_sample: bool = True) -> str:
        """Generate response for a given prompt."""
        
        # Format prompt as conversation
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return ""
    
    def calculate_rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        try:
            scores = self.rouge_scorer.score(reference, generated)
            return {
                'rouge1_f': scores['rouge1'].fmeasure,
                'rouge1_p': scores['rouge1'].precision,
                'rouge1_r': scores['rouge1'].recall,
                'rouge2_f': scores['rouge2'].fmeasure,
                'rouge2_p': scores['rouge2'].precision,
                'rouge2_r': scores['rouge2'].recall,
                'rougeL_f': scores['rougeL'].fmeasure,
                'rougeL_p': scores['rougeL'].precision,
                'rougeL_r': scores['rougeL'].recall
            }
        except Exception as e:
            self.logger.warning(f"ROUGE calculation failed: {e}")
            return {}
    
    def calculate_bleu_score(self, generated: str, reference: str) -> float:
        """Calculate BLEU score."""
        try:
            generated_tokens = word_tokenize(generated.lower())
            reference_tokens = [word_tokenize(reference.lower())]
            
            return sentence_bleu(
                reference_tokens, 
                generated_tokens, 
                smoothing_function=self.bleu_smoother
            )
        except Exception as e:
            self.logger.warning(f"BLEU calculation failed: {e}")
            return 0.0
    
    def calculate_bert_score(self, generated: List[str], reference: List[str]) -> Dict[str, float]:
        """Calculate BERTScore (batch processing for efficiency)."""
        try:
            P, R, F1 = bert_score(generated, reference, lang="en", verbose=False)
            return {
                'bert_precision': P.mean().item(),
                'bert_recall': R.mean().item(),
                'bert_f1': F1.mean().item()
            }
        except Exception as e:
            self.logger.warning(f"BERTScore calculation failed: {e}")
            return {}
    
    def evaluate_conversation(self, 
                            prompt: str, 
                            reference: str, 
                            conversation_id: str,
                            metadata: Optional[Dict[str, Any]] = None) -> ConversationEvaluation:
        """Evaluate a single conversation."""
        
        # Generate response
        generated = self.generate_response(prompt)
        
        # Calculate metrics
        metrics = []
        
        # Standard NLG metrics
        rouge_scores = self.calculate_rouge_scores(generated, reference)
        for metric, score in rouge_scores.items():
            metrics.append(EvaluationResult(metric, score))
        
        # BLEU score
        bleu_score = self.calculate_bleu_score(generated, reference)
        metrics.append(EvaluationResult("bleu", bleu_score))
        
        # Therapeutic quality metrics
        therapeutic_scores = self.therapeutic_metrics.calculate_therapeutic_quality_score(generated)
        for metric, score in therapeutic_scores.items():
            metrics.append(EvaluationResult(f"therapeutic_{metric}", score))
        
        return ConversationEvaluation(
            conversation_id=conversation_id,
            metrics=metrics,
            generated_response=generated,
            reference_response=reference,
            prompt=prompt,
            metadata=metadata
        )
    
    def evaluate_dataset(self, 
                        eval_dataset: Dataset,
                        output_dir: Optional[str] = None,
                        batch_size: int = 8) -> Dict[str, Any]:
        """Evaluate model on a complete dataset."""
        
        self.logger.info(f"Starting evaluation on {len(eval_dataset)} samples")
        
        # Prepare data
        prompts = []
        references = []
        conversation_ids = []
        metadata_list = []
        
        for i, sample in enumerate(eval_dataset):
            # Extract prompt and reference from the sample
            # Assuming the dataset has 'text' field with chat format
            if 'text' in sample:
                # Parse chat format to extract prompt and response
                text = sample['text']
                # Simple parsing - in practice, you'd want more robust parsing
                if '<|im_start|>user' in text and '<|im_start|>assistant' in text:
                    parts = text.split('<|im_start|>assistant')
                    if len(parts) >= 2:
                        prompt_part = parts[0].replace('<|im_start|>user\n', '').strip()
                        response_part = parts[1].split('<|im_end|>')[0].strip()
                        
                        prompts.append(prompt_part)
                        references.append(response_part)
                        conversation_ids.append(sample.get('conversation_id', f'eval_{i}'))
                        metadata_list.append({
                            'source': sample.get('source', 'unknown'),
                            'length': sample.get('length', 0)
                        })
        
        if not prompts:
            self.logger.error("No valid conversations found in dataset")
            return {}
        
        self.logger.info(f"Extracted {len(prompts)} valid conversations")
        
        # Evaluate conversations
        conversation_evaluations = []
        generated_responses = []
        
        for i, (prompt, reference, conv_id, metadata) in enumerate(
            zip(prompts, references, conversation_ids, metadata_list)
        ):
            if i % 10 == 0:
                self.logger.info(f"Evaluating conversation {i+1}/{len(prompts)}")
            
            conv_eval = self.evaluate_conversation(prompt, reference, conv_id, metadata)
            conversation_evaluations.append(conv_eval)
            generated_responses.append(conv_eval.generated_response)
        
        # Calculate batch metrics (like BERTScore)
        bert_scores = self.calculate_bert_score(generated_responses, references)
        
        # Aggregate results
        results = self._aggregate_evaluation_results(conversation_evaluations, bert_scores)
        
        # Save results if output directory provided
        if output_dir:
            self._save_evaluation_results(results, conversation_evaluations, output_dir)
        
        self.logger.info("Evaluation completed")
        return results
    
    def _aggregate_evaluation_results(self, 
                                    conversation_evaluations: List[ConversationEvaluation],
                                    bert_scores: Dict[str, float]) -> Dict[str, Any]:
        """Aggregate evaluation results across all conversations."""
        
        # Collect all metrics
        metric_scores = defaultdict(list)
        
        for conv_eval in conversation_evaluations:
            for metric in conv_eval.metrics:
                metric_scores[metric.metric_name].append(metric.score)
        
        # Calculate aggregated statistics
        aggregated_results = {}
        
        for metric_name, scores in metric_scores.items():
            aggregated_results[metric_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores)
            }
        
        # Add batch metrics
        for metric_name, score in bert_scores.items():
            aggregated_results[metric_name] = {
                'mean': score,
                'std': 0.0,
                'min': score,
                'max': score,
                'median': score
            }
        
        # Calculate composite scores
        therapeutic_metrics = [k for k in aggregated_results.keys() if k.startswith('therapeutic_')]
        if therapeutic_metrics:
            therapeutic_scores = [aggregated_results[m]['mean'] for m in therapeutic_metrics]
            aggregated_results['therapeutic_composite'] = {
                'mean': np.mean(therapeutic_scores),
                'std': np.std(therapeutic_scores),
                'min': np.min(therapeutic_scores),
                'max': np.max(therapeutic_scores),
                'median': np.median(therapeutic_scores)
            }
        
        # Overall quality score (combination of ROUGE-L and therapeutic composite)
        if 'rougeL_f' in aggregated_results and 'therapeutic_composite' in aggregated_results:
            rouge_score = aggregated_results['rougeL_f']['mean']
            therapeutic_score = aggregated_results['therapeutic_composite']['mean']
            overall_score = 0.6 * rouge_score + 0.4 * therapeutic_score
            
            aggregated_results['overall_quality'] = {
                'mean': overall_score,
                'std': 0.0,
                'min': overall_score,
                'max': overall_score,
                'median': overall_score
            }
        
        return aggregated_results
    
    def _save_evaluation_results(self, 
                               results: Dict[str, Any],
                               conversation_evaluations: List[ConversationEvaluation],
                               output_dir: str):
        """Save evaluation results to files."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save aggregated results
        with open(output_path / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save detailed conversation results
        detailed_results = []
        for conv_eval in conversation_evaluations:
            conv_result = {
                'conversation_id': conv_eval.conversation_id,
                'prompt': conv_eval.prompt,
                'reference_response': conv_eval.reference_response,
                'generated_response': conv_eval.generated_response,
                'metadata': conv_eval.metadata,
                'metrics': {m.metric_name: m.score for m in conv_eval.metrics}
            }
            detailed_results.append(conv_result)
        
        with open(output_path / 'detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # Create summary report
        self._create_evaluation_report(results, output_path)
        
        self.logger.info(f"Evaluation results saved to: {output_path}")
    
    def _create_evaluation_report(self, results: Dict[str, Any], output_path: Path):
        """Create a human-readable evaluation report."""
        
        report_lines = [
            "# Therapeutic Conversation Model Evaluation Report",
            f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {self.model_path}",
            "",
            "## Summary Metrics",
            ""
        ]
        
        # Key metrics summary
        key_metrics = [
            'overall_quality', 'rougeL_f', 'bleu', 'bert_f1', 
            'therapeutic_composite', 'therapeutic_empathy', 'therapeutic_reflection'
        ]
        
        for metric in key_metrics:
            if metric in results:
                score = results[metric]['mean']
                report_lines.append(f"- **{metric.replace('_', ' ').title()}**: {score:.4f}")
        
        report_lines.extend([
            "",
            "## Detailed Metrics",
            ""
        ])
        
        # Detailed metrics table
        for metric_name, stats in sorted(results.items()):
            report_lines.extend([
                f"### {metric_name.replace('_', ' ').title()}",
                f"- Mean: {stats['mean']:.4f}",
                f"- Std: {stats['std']:.4f}",
                f"- Min: {stats['min']:.4f}",
                f"- Max: {stats['max']:.4f}",
                f"- Median: {stats['median']:.4f}",
                ""
            ])
        
        # Save report
        with open(output_path / 'evaluation_report.md', 'w') as f:
            f.write('\n'.join(report_lines))
    
    def compare_models(self, 
                      other_evaluator: 'ModelEvaluator',
                      eval_dataset: Dataset,
                      output_dir: str) -> Dict[str, Any]:
        """Compare this model with another model."""
        
        self.logger.info("Starting model comparison")
        
        # Evaluate both models
        results_1 = self.evaluate_dataset(eval_dataset)
        results_2 = other_evaluator.evaluate_dataset(eval_dataset)
        
        # Compare results
        comparison = {}
        
        for metric in results_1.keys():
            if metric in results_2:
                score_1 = results_1[metric]['mean']
                score_2 = results_2[metric]['mean']
                
                comparison[metric] = {
                    'model_1': score_1,
                    'model_2': score_2,
                    'difference': score_1 - score_2,
                    'improvement': ((score_1 - score_2) / score_2 * 100) if score_2 != 0 else 0
                }
        
        # Save comparison results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / 'model_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        self.logger.info(f"Model comparison saved to: {output_path}")
        return comparison
