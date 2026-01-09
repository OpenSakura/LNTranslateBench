#!/usr/bin/env python3
"""
LinguaGacha Output Wrapper for LLM Translate Benchmark

This script takes raw text files (e.g., from LinguaGacha) and wraps them with the
metadata header required by the Stage 2 comparison tool.

Usage:
    python utils/wrap_lingua_gacha.py --input-dir raw_translations --output-dir translated_results --model "Model-Name"
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

def wrap_file(input_path, output_dir, model, source_lang="ja", target_lang="zh"):
    """Reads a raw translation file and writes it with the required metadata header."""
    
    # Read raw content
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        return False

    # Construct output filename
    # Stage 2 expects: sample_name.model_name.translated.txt
    # We assume input filename is "sample_name.txt" or similar
    sample_name = input_path.stem
    output_filename = f"{sample_name}.{model.replace('/', '_')}.translated.txt"
    output_path = output_dir / output_filename
    
    timestamp = datetime.now().isoformat()
    
    # Write wrapped content
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write metadata header
        f.write("=" * 80 + "\n")
        f.write("TRANSLATION METADATA\n")
        f.write("=" * 80 + "\n")
        f.write(f"Sample File: {sample_name}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Source Language: {source_lang}\n")
        f.write(f"Target Language: {target_lang}\n")
        f.write(f"Original Text Length: 0 characters\n") # Unknown, but field required
        f.write(f"Success: True\n")
        f.write(f"Translated Text Length: {len(content)} characters\n")
        f.write(f"Status Code: 200\n")
        f.write("=" * 80 + "\n\n")

        # Write separator
        f.write("=" * 80 + "\n")
        f.write("TRANSLATED TEXT\n")
        f.write("=" * 80 + "\n\n")

        # Write translated text
        f.write(content)
        
    print(f"Wrapped {input_path.name} -> {output_filename}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Wrap LinguaGacha output for benchmark comparison.")
    parser.add_argument("--input-dir", required=True, help="Directory containing raw translated text files")
    parser.add_argument("--output-dir", required=True, help="Directory to save wrapped files")
    parser.add_argument("--model", required=True, help="Model name to embed in metadata")
    parser.add_argument("--source", default="ja", help="Source language code")
    parser.add_argument("--target", default="zh", help="Target language code")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        return
        
    files = list(input_dir.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {input_dir}")
        return
        
    print(f"Found {len(files)} files to process...")
    success_count = 0
    for file_path in files:
        if wrap_file(file_path, output_dir, args.model, args.source, args.target):
            success_count += 1
            
    print(f"Done. Successfully wrapped {success_count}/{len(files)} files.")

if __name__ == "__main__":
    main()
