# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer

local_rank = None

def get_rank():
    """Get the rank of the current process safely."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0

def is_main_process():
    """Check if this is the main process."""
    return get_rank() == 0

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    print(f"Raw predictions shape: {predictions.shape}")
    print(f"Raw labels shape: {labels.shape}")
    print(f"Predictions dtype: {predictions.dtype}")
    print(f"Labels dtype: {labels.dtype}")
    
    # Check for different prediction formats
    if predictions.ndim > 2:
        print("Multi-dimensional predictions - taking argmax on last axis")
        predictions = np.argmax(predictions, axis=-1)
    elif predictions.ndim == 2:
        print("2D predictions - taking argmax on last axis")
        predictions = np.argmax(predictions, axis=-1)
    
    print(f"Predictions after argmax shape: {predictions.shape}")
    print(f"Sample predictions: {predictions.flatten()[:20]}")
    print(f"Sample labels: {labels.flatten()[:20]}")
    
    # Check for padding tokens
    mask = labels != -100
    print(f"Total elements: {len(labels.flatten())}")
    print(f"Valid elements (not -100): {mask.sum()}")
    print(f"Percentage of valid elements: {mask.sum() / len(labels.flatten()) * 100:.2f}%")
    
    # Apply mask
    predictions_filtered = predictions[mask]
    labels_filtered = labels[mask]
    
    print(f"Filtered predictions shape: {predictions_filtered.shape}")
    print(f"Filtered labels shape: {labels_filtered.shape}")
    print(f"Unique prediction values: {np.unique(predictions_filtered)}")
    print(f"Unique label values: {np.unique(labels_filtered)}")
    
    # Calculate accuracy
    if len(predictions_filtered) > 0:
        accuracy = (predictions_filtered == labels_filtered).mean()
        print(f"Matches: {(predictions_filtered == labels_filtered).sum()}")
        print(f"Total valid samples: {len(predictions_filtered)}")
    else:
        accuracy = 0.0
        print("No valid samples found!")
    
    print(f"Final accuracy: {accuracy}")
    print("-" * 50)
    
    return {"accuracy": accuracy}

def run_inference_on_eval_dataset(model, tokenizer, data_module, output_dir, training_args):
    """Run inference on evaluation dataset and save detailed results."""
    
    print("Starting post-training inference on evaluation dataset...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Get the evaluation dataset
    eval_dataset = data_module.get('eval_dataset')
    if eval_dataset is None:
        print("No evaluation dataset found. Skipping inference.")
        return
    
    # Clear GPU cache before inference
    torch.cuda.empty_cache()
    
    print("Using memory-efficient individual sample inference...")
    
    # Process samples individually to avoid memory issues
    all_predictions = []
    all_labels = []
    sample_results = []
    
    total_correct = 0
    total_valid_tokens = 0
    
    with torch.no_grad():
        for i, sample in enumerate(eval_dataset):
            try:
                # Clear cache periodically
                if i % 10 == 0:
                    torch.cuda.empty_cache()
                
                # Prepare inputs
                inputs = {}
                for key, value in sample.items():
                    if key != 'labels':
                        if isinstance(value, torch.Tensor):
                            inputs[key] = value.unsqueeze(0).to(model.device)
                        elif isinstance(value, (list, np.ndarray)):
                            inputs[key] = torch.tensor(value).unsqueeze(0).to(model.device)
                        else:
                            inputs[key] = torch.tensor([value]).to(model.device)
                
                # Get model outputs
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Get predictions
                pred_tokens = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
                
                # Get labels
                if 'labels' in sample:
                    labels_tensor = sample['labels']
                    if isinstance(labels_tensor, torch.Tensor):
                        labels_array = labels_tensor.cpu().numpy()
                    else:
                        labels_array = np.array(labels_tensor)
                    
                    # Calculate accuracy for this sample
                    mask = labels_array != -100
                    if mask.sum() > 0:
                        sample_correct = (pred_tokens[mask] == labels_array[mask]).sum()
                        sample_valid = mask.sum()
                        sample_accuracy = sample_correct / sample_valid
                        
                        total_correct += sample_correct
                        total_valid_tokens += sample_valid
                    else:
                        sample_accuracy = 0.0
                        sample_valid = 0
                    
                    # Decode predictions and labels to text
                    try:
                        pred_text = tokenizer.decode(pred_tokens[mask], skip_special_tokens=True)
                        label_text = tokenizer.decode(labels_array[mask], skip_special_tokens=True)
                    except:
                        pred_text = "Error decoding prediction"
                        label_text = "Error decoding label"
                    
                    # Store sample result
                    sample_result = {
                        'sample_id': i,
                        'accuracy': float(sample_accuracy),
                        'valid_tokens': int(sample_valid),
                        'predicted_text': pred_text,
                        'target_text': label_text,
                        'predicted_tokens': pred_tokens[mask].tolist() if mask.sum() > 0 else [],
                        'target_tokens': labels_array[mask].tolist() if mask.sum() > 0 else [],
                    }
                    
                    # Add input information if available
                    if 'input_ids' in sample:
                        try:
                            input_ids = sample['input_ids']
                            if isinstance(input_ids, torch.Tensor):
                                input_ids = input_ids.cpu().numpy()
                            input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                            sample_result['input_text'] = input_text
                        except:
                            sample_result['input_text'] = "Error decoding input"
                    
                    sample_results.append(sample_result)
                
                if i % 10 == 0:
                    print(f"Processed {i+1}/{len(eval_dataset)} samples")
                    
            except Exception as sample_error:
                print(f"Error processing sample {i}: {sample_error}")
                continue
    
    # Calculate overall accuracy
    if total_valid_tokens > 0:
        overall_accuracy = total_correct / total_valid_tokens
    else:
        overall_accuracy = 0.0
    
    eval_results = {'eval_accuracy': overall_accuracy}
    
    # Prepare detailed results
    detailed_results = {
        'overall_accuracy': eval_results.get('eval_accuracy', 0.0),
        'total_samples': len(eval_dataset),
        'total_valid_tokens': int(total_valid_tokens),
        'total_correct_tokens': int(total_correct),
        'samples': sample_results
    }
    
    # Save detailed results
    results_file = os.path.join(output_dir, 'detailed_inference_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("INFERENCE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {detailed_results['overall_accuracy']:.4f}")
    print(f"Total Samples: {detailed_results['total_samples']}")
    print(f"Total Valid Tokens: {detailed_results['total_valid_tokens']}")
    print(f"Total Correct Tokens: {detailed_results['total_correct_tokens']}")
    print(f"Results saved to: {results_file}")
    
    # Print sample-by-sample accuracy distribution
    if sample_results:
        sample_accuracies = [s['accuracy'] for s in sample_results]
        print(f"\nSample Accuracy Distribution:")
        print(f"  Mean: {np.mean(sample_accuracies):.4f}")
        print(f"  Std:  {np.std(sample_accuracies):.4f}")
        print(f"  Min:  {np.min(sample_accuracies):.4f}")
        print(f"  Max:  {np.max(sample_accuracies):.4f}")
        
        # Show some examples
        print(f"\nFirst 3 Sample Results:")
        for i, sample in enumerate(sample_results[:3]):
            print(f"  Sample {i+1} (ID: {sample['sample_id']}):")
            print(f"    Accuracy: {sample['accuracy']:.4f}")
            print(f"    Valid tokens: {sample['valid_tokens']}")
            print(f"    Predicted: {sample['predicted_text'][:100]}...")
            print(f"    Target:    {sample['target_text'][:100]}...")
            print()
    
    print(f"{'='*60}")
    
    # Clear GPU cache after inference
    torch.cuda.empty_cache()
    
    return detailed_results

def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    if is_main_process():
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()
    
    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, compute_metrics=compute_metrics, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    # Run inference on evaluation dataset after training
    if is_main_process():
        print("\n" + "="*60)
        print("STARTING POST-TRAINING INFERENCE")
        print("="*60)
        
        inference_results = run_inference_on_eval_dataset(
            model=model, 
            tokenizer=tokenizer, 
            data_module=data_module,
            output_dir=training_args.output_dir,
            training_args=training_args
        )
        
        print("Post-training inference completed!")
        print("="*60)

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")