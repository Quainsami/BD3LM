# scripts/precompute_unigrams.py
import sys
import os
# Add project root to sys.path if running script from project root or scripts/
# Ensure this path adjustment is correct for your execution environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from collections import Counter
from tqdm import tqdm
import fsspec
import datasets

# Assuming dataloader.get_tokenizer and utils.fsspec_exists are available
import dataloader as project_dataloader # aliased to avoid conflict
import utils as project_utils

def compute_and_save_unigram_log_probs(dataset, tokenizer, save_path, vocab_size, text_column_name='text', tokenization_batch_size=1000):
    """
    Computes unigram frequencies from a Hugging Face dataset and saves log probabilities.
    """
    if project_utils.fsspec_exists(save_path):
        print(f"Unigram log probabilities already exist at {save_path}, skipping computation.")
        # Ensure loading to CPU first to avoid device mismatches if script is run on CPU-only machine
        return torch.load(save_path, map_location='cpu')

    print(f"Computing unigram frequencies for dataset (batch_size={tokenization_batch_size}), will save to {save_path}...")
    counts = Counter()
    total_tokens = 0
    processed_examples = 0 # To count how many items from the dataset are actually processed

    try:
        num_examples = len(dataset)
    except TypeError:
        num_examples = None
        print("Warning: Dataset length not available (streaming dataset?). TQDM progress may be inaccurate.")

    # Buffer for batching texts before tokenization
    text_batch_buffer = []

    for example in tqdm(dataset, total=num_examples, desc="Reading dataset for unigrams"):
        text_content = example.get(text_column_name)
        if text_content is None:
            # Optionally, log this warning less frequently or to a file if it's too noisy
            # print(f"Warning: Text column '{text_column_name}' not found in example: {list(example.keys())}. Skipping example.")
            continue

        current_example_texts = []
        if isinstance(text_content, list):
            # Filter out None or empty strings from the list of texts
            current_example_texts.extend(item for item in text_content if isinstance(item, str) and item)
        elif isinstance(text_content, str):
            if text_content: # Ensure the string is not empty
                current_example_texts.append(text_content)
        # else:
            # Optionally, log this warning
            # print(f"Warning: Skipping example with unexpected text_column type: {type(text_content)}")
        
        text_batch_buffer.extend(current_example_texts)
        processed_examples += 1 

        if len(text_batch_buffer) >= tokenization_batch_size:
            # Process the current batch
            # Use tokenizer() for batch encoding, which is faster
            # Set truncation to False because we want all tokens for unigram counts.
            # Padding is not needed here.
            if text_batch_buffer: # Ensure buffer is not empty before tokenizing
                encoded_batch = tokenizer(text_batch_buffer, add_special_tokens=False, truncation=False, padding=False)
                for token_ids_one_doc in encoded_batch["input_ids"]:
                    counts.update(token_ids_one_doc)
                    total_tokens += len(token_ids_one_doc)
            text_batch_buffer = [] # Reset buffer

    # Process any remaining texts in the buffer
    if text_batch_buffer:
        encoded_batch = tokenizer(text_batch_buffer, add_special_tokens=False, truncation=False, padding=False)
        for token_ids_one_doc in encoded_batch["input_ids"]:
            counts.update(token_ids_one_doc)
            total_tokens += len(token_ids_one_doc)
    
    print(f"Processed {processed_examples} examples, found {total_tokens} total tokens after tokenization.")

    if total_tokens == 0:
        print("Warning: No tokens processed from the dataset. Creating uniform log probabilities.")
        unigram_log_probs = torch.full((vocab_size,), -torch.log(torch.tensor(vocab_size, dtype=torch.float32)))
    else:
        unigram_probs = torch.zeros(vocab_size, dtype=torch.float32)
        for token_id, count in counts.items():
            if 0 <= token_id < vocab_size: # Ensure token_id is within bounds
                unigram_probs[token_id] = count
        
        unigram_probs = unigram_probs / total_tokens
        epsilon = 1e-10 # Adjusted epsilon for numerical stability with log
        unigram_probs.clamp_(min=epsilon) # Clamp probabilities before log
        unigram_log_probs = torch.log(unigram_probs)

    save_dir = os.path.dirname(save_path)
    if save_dir and not project_utils.fsspec_exists(save_dir): # Check if save_dir is not empty string
        project_utils.fsspec_mkdirs(save_dir, exist_ok=True)
    
    # Ensure parent directory of the file itself exists
    if os.path.dirname(save_path) and not project_utils.fsspec_exists(os.path.dirname(save_path)):
        project_utils.fsspec_mkdirs(os.path.dirname(save_path), exist_ok=True)

    with fsspec.open(save_path, 'wb') as f:
        torch.save(unigram_log_probs, f)

    print(f"Saved unigram log probabilities to {save_path}")
    return unigram_log_probs

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Starting unigram frequency precomputation using Hydra config...")
    print(f"Effective configuration: \n{OmegaConf.to_yaml(cfg)}")
    
    tokenizer = project_dataloader.get_tokenizer(cfg)
    
    dataset_identifier_to_load = cfg.data.train 
    dataset_split_for_unigrams = 'train' 

    print(f"Loading dataset for unigrams: '{dataset_identifier_to_load}' (split: '{dataset_split_for_unigrams}') based on cfg.data.train")

    # Specific dataset loading logic
    if dataset_identifier_to_load == 'text8':
        dataset_dict = project_dataloader.get_text8_dataset(cache_dir=cfg.data.cache_dir, max_seq_length=cfg.model.length)
        raw_train_dataset = dataset_dict[dataset_split_for_unigrams]
        text_column = 'text'
    elif dataset_identifier_to_load == 'openwebtext':
        raw_train_dataset = datasets.load_dataset('openwebtext', split=dataset_split_for_unigrams, cache_dir=cfg.data.cache_dir, trust_remote_code=True)
        text_column = 'text'
    elif dataset_identifier_to_load == 'wikitext103':
        raw_train_dataset = datasets.load_dataset('wikitext', 'wikitext-103-raw-v1', split=dataset_split_for_unigrams, cache_dir=cfg.data.cache_dir, trust_remote_code=True)
        text_column = 'text'
    elif dataset_identifier_to_load == 'wikitext2':
        raw_train_dataset = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split=dataset_split_for_unigrams, cache_dir=cfg.data.cache_dir, trust_remote_code=True)
        text_column = 'text'
    elif dataset_identifier_to_load == 'ptb':
        raw_train_dataset = datasets.load_dataset('ptb_text_only', split=dataset_split_for_unigrams, cache_dir=cfg.data.cache_dir, trust_remote_code=True)
        text_column = 'sentence'
    elif dataset_identifier_to_load == 'scientific_papers_arxiv':
       raw_train_dataset = datasets.load_dataset('scientific_papers', 'arxiv', split=dataset_split_for_unigrams, cache_dir=cfg.data.cache_dir, trust_remote_code=True)
       text_column = 'article'
    elif dataset_identifier_to_load == 'scientific_papers_pubmed':
       raw_train_dataset = datasets.load_dataset('scientific_papers', 'pubmed', split=dataset_split_for_unigrams, cache_dir=cfg.data.cache_dir, trust_remote_code=True)
       text_column = 'article'
    elif dataset_identifier_to_load == 'ag_news':
        raw_train_dataset = datasets.load_dataset('ag_news', split=dataset_split_for_unigrams, cache_dir=cfg.data.cache_dir, trust_remote_code=True)
        text_column = 'text' # AG News has 'text' and 'label'
    else:
        # General case for other datasets specified in cfg.data.train
        print(f"Attempting to load general dataset: {dataset_identifier_to_load}")
        raw_train_dataset = datasets.load_dataset(dataset_identifier_to_load, split=dataset_split_for_unigrams, cache_dir=cfg.data.cache_dir, trust_remote_code=True)
        # Attempt to infer text column (add more robust inference if needed)
        if 'text' in raw_train_dataset.column_names:
            text_column = 'text'
        elif 'sentence' in raw_train_dataset.column_names:
            text_column = 'sentence'
        elif 'article' in raw_train_dataset.column_names:
            text_column = 'article'
        else: # Fallback to the first column if common names aren't found
            text_column = raw_train_dataset.column_names[0]
            print(f"Warning: Auto-detected text column as '{text_column}'. Verify this is correct.")

    # Construct filename based on the dataset identifier from cfg.data.train
    unigram_log_probs_filename = f"unigram_log_probs_{dataset_identifier_to_load}_{tokenizer.name_or_path.replace('/', '_')}.pt"
    unigram_log_probs_path = os.path.join(
        getattr(cfg.data, 'cache_dir', './cache'), # Use data.cache_dir from config
        unigram_log_probs_filename
    )
    
    print(f"Target save path for unigrams: {unigram_log_probs_path}")

    _ = compute_and_save_unigram_log_probs(
        dataset=raw_train_dataset,
        tokenizer=tokenizer,
        save_path=unigram_log_probs_path,
        vocab_size=tokenizer.vocab_size,
        text_column_name=text_column
        # tokenization_batch_size can be added here if you want to configure it via Hydra
    )
    print(f"Unigram log probabilities computation complete. Saved to {unigram_log_probs_path}")

if __name__ == "__main__":
    main()