import os
import sys

print("=" * 60)
print("ALTERNATIVE DATASET DOWNLOAD SCRIPT")
print("=" * 60)

# Alternative 1: SQuAD
print("\n[1/3] Attempting SQuAD dataset...")
try:
    from datasets import load_dataset
    print("  Loading SQuAD...")
    squad = load_dataset('squad', split='train[:5000]')  # First 5000 for testing
    print(f"  ✓ SQuAD loaded! {len(squad)} examples")
    
    # Save to CSV
    import csv
    with open('data/squad_conversations.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['question', 'answer', 'context'])
        for example in squad:
            writer.writerow([example['question'], example['answers']['text'][0] if example['answers']['text'] else '', example['context'][:200]])
    print("  ✓ Saved to data/squad_conversations.csv")
except Exception as e:
    print(f"  ✗ SQuAD failed: {str(e)[:100]}")

# Alternative 2: MS MARCO (smaller subset)
print("\n[2/3] Attempting MS MARCO dataset...")
try:
    ms_marco = load_dataset('ms_marco', 'v2.1', split='train[:2000]')
    print(f"  ✓ MS MARCO loaded! {len(ms_marco)} examples")
    
    with open('data/ms_marco_conversations.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['query', 'answer'])
        for example in ms_marco:
            writer.writerow([example['query'], str(example['answers'])[:200]])
    print("  ✓ Saved to data/ms_marco_conversations.csv")
except Exception as e:
    print(f"  ✗ MS MARCO failed: {str(e)[:100]}")

# Alternative 3: Use Cornell (Local)
print("\n[3/3] Attempting Cornell Movie-Dialogs (Local)...")
try:
    import pandas as pd
    cornell_path = 'data/archive/movie_lines.tsv'
    if os.path.exists(cornell_path):
        df = pd.read_csv(cornell_path, sep='\t', header=None, nrows=5000)
        df.columns = ['line_id', 'character_id', 'movie_id', 'character', 'text']
        
        with open('data/cornell_conversations.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['line_id', 'character', 'text'])
            for idx, row in df.iterrows():
                writer.writerow([row['line_id'], row['character'], row['text']])
        print(f"  ✓ Cornell loaded! {len(df)} lines")
        print("  ✓ Saved to data/cornell_conversations.csv")
    else:
        print(f"  ✗ Cornell data not found at {cornell_path}")
except Exception as e:
    print(f"  ✗ Cornell failed: {str(e)[:100]}")

print("\n" + "=" * 60)
print("SUMMARY: Check data/ folder for CSV files")
print("=" * 60)
