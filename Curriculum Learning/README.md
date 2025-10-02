# Gemma 3n Therapeutic Conversation Fine-tuning

A professional-grade fine-tuning pipeline for Google's Gemma 3n E2B model, specialized for therapeutic conversations using CounselChat and AnnoMI datasets.

## 🎯 Project Overview

This project implements a sophisticated three-stage fine-tuning approach:

1. **Stage 1 (SFT)**: Fine-tune on CounselChat for general therapeutic conversation patterns
2. **Stage 2 (SFT)**: Specialize on AnnoMI for motivational interviewing techniques  
3. **Stage 3 (DPO)**: Apply preference optimization for improved response quality and alignment

## 🏗️ Architecture

### Sequential Training (Original)
```
Base Model: google/gemma-3n-E2B-it
     ↓
Stage 1: CounselChat SFT (General Therapy)
     ↓
Stage 2: AnnoMI SFT (Motivational Interviewing)
     ↓
Stage 3: DPO (Preference Optimization)
     ↓
Final Model: Merged & Deployment Ready
```

### Curriculum Learning (RECOMMENDED)
```
Base Model: google/gemma-3n-E2B-it
     ↓
Phase 1: 100% CounselChat (Foundation)
     ↓
Phase 2: 80% CounselChat + 20% AnnoMI (Introduction)
     ↓
Phase 3: 60% CounselChat + 40% AnnoMI (Integration)
     ↓
Phase 4: 40% CounselChat + 60% AnnoMI (Specialization)
     ↓
Phase 5: 100% AnnoMI (Mastery)
     ↓
DPO: Preference Optimization (Optional)
     ↓
Final Model: Merged & Deployment Ready
```

## 📊 Datasets

### CounselChat
- **Source**: `nbertagnolli/counsel-chat` (HuggingFace)
- **Purpose**: General therapeutic conversation patterns
- **Format**: Question-answer pairs from mental health counseling
- **Size**: ~3,500 single-turn conversations
- **Characteristics**: Breadth of therapeutic topics, shorter interactions

### AnnoMI
- **Source**: Local CSV file (`AnnoMI-full.csv`)
- **Purpose**: Specialized motivational interviewing techniques
- **Format**: Annotated therapeutic conversations with MI quality ratings
- **Size**: 133 multi-turn conversations (~102 turns per conversation)
- **Characteristics**: Deep conversational patterns, MI technique annotations
- **Features**: Therapist behaviors, client talk types, MI quality scores

### Dataset Complementarity
The datasets are **highly complementary**:
- **CounselChat**: Provides breadth (many topics, diverse scenarios)
- **AnnoMI**: Provides depth (extended conversations, specialized techniques)
- **Volume Ratio**: 26:1 (CounselChat:AnnoMI conversations)
- **Structure Difference**: Single-turn vs multi-turn conversations

This significant imbalance and structural difference makes **curriculum learning the optimal approach**.

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure you have the AnnoMI dataset
# Place AnnoMI-full.csv in the project root
```

### Complete Pipeline

```bash
# Option 1: Sequential Training (Original approach)
python scripts/train_complete_pipeline.py \
    --output_dir ./checkpoints \
    --experiment_name gemma3n_therapeutic \
    --debug  # Remove for full training

# Option 2: Curriculum Learning (RECOMMENDED for your datasets)
python scripts/train_curriculum.py \
    --output_dir ./checkpoints/curriculum \
    --experiment_name gemma3n_curriculum \
    --curriculum_strategy therapeutic \
    --run_dpo_after \
    --merge_final_model \
    --debug  # Remove for full training
```

### Individual Stages

```bash
# Sequential Training Approach
# Stage 1: CounselChat SFT
python scripts/train_stage1.py \
    --output_dir ./checkpoints/stage1 \
    --experiment_name stage1_counselchat

# Stage 2: AnnoMI SFT
python scripts/train_stage2.py \
    --output_dir ./checkpoints/stage2 \
    --stage1_model_path ./checkpoints/stage1/[model_dir] \
    --experiment_name stage2_annomi

# Stage 3: DPO
python scripts/preference_tuning.py \
    --output_dir ./checkpoints/dpo \
    --sft_model_path ./checkpoints/stage2/[model_dir] \
    --experiment_name dpo_tuning

# Curriculum Learning Approach (RECOMMENDED)
python scripts/train_curriculum.py \
    --curriculum_strategy therapeutic \
    --target_dataset_size 2000 \
    --quality_filter high

# Merge adapters for deployment
python scripts/merge_adapters.py \
    --adapter_path ./checkpoints/dpo/[model_dir] \
    --output_dir ./final_model \
    --test_generation
```

## ⚙️ Configuration

### Model Configuration (`configs/model_config.yaml`)
```yaml
model:
  name: "google/gemma-3n-E2B-it"
  torch_dtype: "bfloat16"
  device_map: "auto"
  attn_implementation: "flash_attention_2"

tokenizer:
  name: "google/gemma-3n-E2B-it"
  padding_side: "right"
  add_eos_token: true
```

### Training Configuration (`configs/training_config.yaml`)
```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2e-4
  max_seq_length: 2048
  bf16: true
```

### LoRA Configuration (`configs/lora_config.yaml`)
```yaml
lora:
  r: 64
  lora_alpha: 128
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  lora_dropout: 0.1
```

## 🎓 Curriculum Learning

### Why Curriculum Learning?

Based on analysis of your datasets, **curriculum learning is strongly recommended** over sequential training:

1. **Volume Imbalance**: 26:1 ratio between datasets requires careful balancing
2. **Structural Differences**: Single-turn vs multi-turn conversations need gradual integration
3. **Complementary Strengths**: CounselChat (breadth) + AnnoMI (depth) = optimal combination
4. **Catastrophic Forgetting Prevention**: Gradual mixing preserves learned patterns

### Curriculum Phases

| Phase | CounselChat | AnnoMI | Focus | Learning Rate |
|-------|-------------|--------|-------|---------------|
| 1. Foundation | 100% | 0% | Basic therapeutic patterns | 1e-4 |
| 2. Introduction | 80% | 20% | Introduce MI concepts | 8e-5 |
| 3. Integration | 60% | 40% | Balance breadth/depth | 6e-5 |
| 4. Specialization | 40% | 60% | Advanced MI techniques | 4e-5 |
| 5. Mastery | 0% | 100% | Deep MI specialization | 2e-5 |

### Curriculum Benefits

- **Gradual Complexity**: Start simple, increase sophistication
- **Knowledge Preservation**: Maintain CounselChat patterns while learning AnnoMI
- **Optimal Convergence**: Better final performance than sequential training
- **Adaptive Learning Rates**: Decrease as specialization increases

## 📁 Project Structure

```
gemma_3n/
├── configs/                    # Configuration files
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── lora_config.yaml
├── src/                        # Source code
│   ├── utils.py               # Utilities and helpers
│   ├── data_preprocessing.py  # Dataset processing
│   ├── training.py           # Training logic
│   └── eval.py               # Evaluation metrics
├── scripts/                   # Training scripts
│   ├── train_stage1.py       # Stage 1 training
│   ├── train_stage2.py       # Stage 2 training
│   ├── preference_tuning.py  # DPO training
│   └── merge_adapters.py     # Model merging
├── checkpoints/              # Model checkpoints
├── logs/                     # Training logs
├── AnnoMI-full.csv          # AnnoMI dataset
└── requirements.txt         # Dependencies
```

## 🔧 Advanced Usage

### Custom Dataset Processing

```python
from src.data_preprocessing import CounselChatProcessor, AnnoMIProcessor

# Process CounselChat
processor = CounselChatProcessor("nbertagnolli/counsel-chat")
conversations = processor.process_conversations()

# Process AnnoMI with quality filtering
processor = AnnoMIProcessor("./AnnoMI-full.csv")
raw_data = processor.load_raw_data()
high_quality = raw_data[raw_data['mi_quality'] == 'high']
conversations = processor.process_conversations(high_quality)
```

### Custom Training Configuration

```python
from src.training import TrainingPipeline

# Initialize pipeline
pipeline = TrainingPipeline(
    model_config=model_config,
    training_config=training_config,
    lora_config=lora_config
)

# Run custom training
results = pipeline.run_full_pipeline(
    stage1_datasets=(train_ds1, eval_ds1),
    stage2_datasets=(train_ds2, eval_ds2),
    dpo_datasets=(train_dpo, eval_dpo)
)
```

### Model Evaluation

```python
from src.eval import ModelEvaluator

# Evaluate trained model
evaluator = ModelEvaluator("./checkpoints/dpo/final_model")
results = evaluator.evaluate_dataset(eval_dataset, "./evaluation_results")

# Therapeutic-specific metrics included:
# - Empathy score
# - Reflection techniques
# - MI technique usage
# - Response appropriateness
```

## 📈 Monitoring & Logging

### Weights & Biases Integration
```yaml
wandb:
  project: "gemma3n-therapeutic-finetuning"
  tags: ["gemma3n", "therapeutic", "counseling"]
```

### Comprehensive Logging
- Training metrics and loss curves
- Model parameter counts and memory usage
- Dataset statistics and preprocessing info
- Evaluation results with therapeutic metrics

## 🎯 Evaluation Metrics

### Standard NLG Metrics
- ROUGE (1, 2, L)
- BLEU score
- BERTScore

### Therapeutic-Specific Metrics
- **Empathy Score**: Keyword-based empathy detection
- **Reflection Score**: Therapeutic reflection pattern matching
- **MI Techniques**: Open questions, reflections, affirmations, summaries
- **Response Appropriateness**: Length and content quality

### Composite Scores
- **Therapeutic Quality**: Weighted combination of therapeutic metrics
- **Overall Quality**: Balance of NLG and therapeutic metrics

## 🔍 Model Capabilities

The fine-tuned model specializes in:

- **Empathetic Responses**: Understanding and validating client emotions
- **Motivational Interviewing**: Using MI techniques effectively
- **Therapeutic Questioning**: Asking appropriate open-ended questions
- **Reflection Skills**: Paraphrasing and reflecting client statements
- **Professional Boundaries**: Maintaining appropriate therapeutic stance

## ⚡ Performance Optimization

### Memory Optimization
- LoRA for parameter-efficient fine-tuning
- Gradient checkpointing
- Mixed precision training (bfloat16)
- Flash Attention 2 implementation

### Training Efficiency
- Gradient accumulation for effective large batch sizes
- Optimized data loading with multiple workers
- Smart caching and preprocessing

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Enable gradient checkpointing

2. **Dataset Loading Errors**
   - Verify AnnoMI-full.csv is in project root
   - Check HuggingFace dataset access permissions
   - Ensure proper internet connection for dataset download

3. **Model Loading Issues**
   - Verify HuggingFace Hub access to Gemma models
   - Check model path specifications
   - Ensure sufficient disk space for model downloads

### Debug Mode
Use `--debug` flag for reduced dataset sizes and faster iteration during development.

## 📝 Citation

If you use this work, please cite:

```bibtex
@software{gemma3n_therapeutic_finetuning,
  title={Gemma 3n Therapeutic Conversation Fine-tuning},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[your-repo]/gemma3n-therapeutic}
}
```

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

Note: The Gemma model is subject to Google's Gemma License. Please review and comply with the original model's terms of use.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues:
- Open a GitHub issue
- Check the troubleshooting section
- Review the logs in the `logs/` directory

---

**Disclaimer**: This model is for research and educational purposes. It should not be used as a substitute for professional mental health services. Always consult qualified healthcare professionals for therapeutic interventions. 