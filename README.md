# GPT-2 from Scratch: Implementation and Fine-tuning

A project implementing GPT-2 from scratch and demonstrating various fine-tuning techniques including LoRA for spam classification and instruction following.

## Overview

This project implements a complete GPT-2 model from scratch using PyTorch, covering:

- **Model Architecture**: Layer normalization, GELU activation, multi-head attention, transformer blocks
- **Pretrained Weights**: Loading and using OpenAI's GPT-2 weights
- **Spam Classification**: Fine-tuning with LoRA (Low-Rank Adaptation) for SMS spam detection
- **Instruction Following**: Full fine-tuning of GPT-2 Medium on instruction datasets
- **Evaluation**: Using LLM-as-a-Judge with Llama 3 for response quality assessment


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

This project is based on the book **"Build a Large Language Model (From Scratch)"** by **Sebastian Raschka**. The implementation follows the concepts and techniques presented in the book .
