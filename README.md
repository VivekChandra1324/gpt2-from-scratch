# GPT-2 from Scratch: Implementation and Fine-tuning

A learning project implementing GPT-2 from scratch and demonstrating various fine-tuning techniques including LoRA for spam classification and instruction following.

## Overview

This project implements a complete GPT-2 model from scratch using PyTorch, covering:

- **Model Architecture**: Layer normalization, GELU activation, multi-head attention, transformer blocks
- **Pretrained Weights**: Loading and using OpenAI's GPT-2 weights
- **Spam Classification**: Fine-tuning with LoRA (Low-Rank Adaptation) for SMS spam detection
- **Instruction Following**: Full fine-tuning of GPT-2 Medium on instruction datasets
- **Evaluation**: Using LLM-as-a-Judge with Llama 3 for response quality assessment

## Project Structure

```
GPT-2/
├── gpt2-from-scratch.ipynb    # Main implementation notebook
├── gpt_download.py            # GPT-2 weight download utility
├── gpt2/                      # Pretrained model weights
│   ├── 124M/                  # GPT-2 Small (124M parameters)
│   └── 355M/                  # GPT-2 Medium (355M parameters)
├── sms_spam_collection/       # Spam dataset
├── train.csv                  # Training data
├── validation.csv             # Validation data
├── test.csv                   # Test data
├── instruction-data.json      # Instruction dataset
└── instruction-data-with-response.json  # Generated responses
```

## Requirements

- Python 3.8+
- PyTorch
- tiktoken
- pandas
- matplotlib
- tqdm
- psutil (for Ollama integration)

## Usage

1. **Run the notebook**: Open `gpt2-from-scratch.ipynb` and execute cells sequentially
2. **Spam Classification**: The notebook will download the SMS spam dataset and train a LoRA-adapted classifier
3. **Instruction Fine-tuning**: Fine-tune GPT-2 Medium on instruction-following tasks
4. **Evaluation**: Use Ollama with Llama 3 for response quality assessment

## Key Features

### Model Implementation
- Custom LayerNorm, GELU, and MultiHeadAttention implementations
- Complete GPT-2 architecture with transformer blocks
- Support for both GPT-2 Small (124M) and Medium (355M) models

### Fine-tuning Techniques
- **LoRA**: Parameter-efficient fine-tuning for spam classification
- **Full Fine-tuning**: Complete model training for instruction following
- Custom dataset handling and collate functions

### Evaluation
- Training/validation loss and accuracy tracking
- Test set evaluation
- LLM-as-a-Judge scoring with Llama 3

## Results

The implementation demonstrates:
- Successful loading and inference with pretrained GPT-2 weights
- Effective spam classification using LoRA adaptation
- Instruction following capabilities after fine-tuning
- Automated evaluation using modern LLM scoring methods

## Acknowledgments

This project is based on the book **"Build a Large Language Model (From Scratch)"** by **Sebastian Raschka**. The implementation follows the concepts and techniques presented in the book, adapted for educational purposes.

Special thanks to Sebastian Raschka for providing comprehensive guidance on:
- Transformer architecture implementation
- GPT-2 model construction
- Fine-tuning strategies including LoRA
- Modern evaluation techniques

## Learning Objectives

This project serves as a hands-on learning experience for:
- Understanding transformer architecture fundamentals
- Implementing large language models from scratch
- Exploring parameter-efficient fine-tuning methods
- Working with instruction-following datasets
- Evaluating generative model performance

## License

This is a learning project based on educational materials. Please refer to the original book and OpenAI's GPT-2 license for usage terms.
