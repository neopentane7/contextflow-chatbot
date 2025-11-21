import pandas as pd
import os

print("=" * 70)
print("MERGING ALL DATASETS FOR TRAINING")
print("=" * 70)

all_data = []

# 1. SQuAD
print("\n[1/4] Loading SQuAD...")
try:
    squad = pd.read_csv('data/squad_conversations.csv')
    squad_clean = squad[['question', 'answer']].dropna()
    squad_clean.columns = ['input', 'response']
    squad_clean['dataset'] = 'squad'
    all_data.append(squad_clean)
    print(f"  ✓ SQuAD: {len(squad_clean):,} pairs")
except Exception as e:
    print(f"  ✗ SQuAD error: {e}")

# 2. MS MARCO
print("[2/4] Loading MS MARCO...")
try:
    marco = pd.read_csv('data/ms_marco_conversations.csv')
    marco_clean = marco[['query', 'answer']].dropna()
    marco_clean.columns = ['input', 'response']
    marco_clean['dataset'] = 'ms_marco'
    all_data.append(marco_clean)
    print(f"  ✓ MS MARCO: {len(marco_clean):,} pairs")
except Exception as e:
    print(f"  ✗ MS MARCO error: {e}")

# 3. Cornell Movie-Dialogs
print("[3/4] Loading Cornell Movie-Dialogs...")
try:
    # Try multiple possible paths
    cornell_paths = [
        'data/cornell/movie_lines.tsv',
        'data/archive/movie_lines.tsv',
        'data/cornell/movie_lines.txt'
    ]
    
    cornell_path = None
    for path in cornell_paths:
        if os.path.exists(path):
            cornell_path = path
            print(f"  Found Cornell at: {path}")
            break
    
    if cornell_path:
        # Read with proper encoding and error handling
        cornell_lines = pd.read_csv(
            cornell_path, 
            sep='\t',
            header=None, 
            encoding='latin-1',
            on_bad_lines='skip',
            quoting=3  # Ignore quotes
        )
        
        # Assign column names
        cornell_lines.columns = ['line_id', 'char_id', 'movie_id', 'char_name', 'text']
        
        # Remove NaN and empty text
        cornell_lines = cornell_lines.dropna(subset=['text'])
        cornell_lines = cornell_lines[cornell_lines['text'].str.strip() != '']
        
        # Create conversation pairs (consecutive lines)
        inputs = cornell_lines['text'][::2].values
        responses = cornell_lines['text'][1::2].values
        min_len = min(len(inputs), len(responses))
        
        cornell_pairs = pd.DataFrame({
            'input': inputs[:min_len],
            'response': responses[:min_len],
            'dataset': 'cornell'
        })
        
        all_data.append(cornell_pairs)
        print(f"  ✓ Cornell: {len(cornell_pairs):,} pairs")
    else:
        print("  ✗ Cornell not found at any expected path")
        print("  Expected paths:")
        for p in cornell_paths:
            print(f"    - {p}")
except Exception as e:
    print(f"  ✗ Cornell error: {str(e)[:200]}")

# 4. Ubuntu Dialogue Corpus
print("[4/4] Loading Ubuntu Dialogue Corpus...")
try:
    # Common Ubuntu file structures
    ubuntu_paths = [
        'data/ubuntu/dialogueText.csv',
        'data/ubuntu/ubuntu_dialogs.csv',
        'data/ubuntu/train.csv'
    ]
    
    loaded = False
    for path in ubuntu_paths:
        if os.path.exists(path):
            print(f"  Found Ubuntu at: {path}")
            ubuntu = pd.read_csv(path, nrows=5000, encoding='latin-1', on_bad_lines='skip')
            
            # Try to identify correct columns
            if 'Context' in ubuntu.columns and 'Utterance' in ubuntu.columns:
                ubuntu_pairs = ubuntu[['Context', 'Utterance']].dropna()
                ubuntu_pairs.columns = ['input', 'response']
            elif 'question' in ubuntu.columns and 'answer' in ubuntu.columns:
                ubuntu_pairs = ubuntu[['question', 'answer']].dropna()
                ubuntu_pairs.columns = ['input', 'response']
            else:
                # Take first two text columns
                text_cols = ubuntu.select_dtypes(include=['object']).columns[:2]
                if len(text_cols) >= 2:
                    ubuntu_pairs = ubuntu[text_cols].dropna()
                    ubuntu_pairs.columns = ['input', 'response']
                else:
                    print(f"  ⚠ Can't identify Ubuntu columns: {ubuntu.columns.tolist()}")
                    continue
            
            ubuntu_pairs['dataset'] = 'ubuntu'
            all_data.append(ubuntu_pairs)
            print(f"  ✓ Ubuntu: {len(ubuntu_pairs):,} pairs")
            loaded = True
            break
    
    if not loaded:
        print("  ⚠ Ubuntu: No compatible files found")
except Exception as e:
    print(f"  ✗ Ubuntu error: {str(e)[:200]}")

# Merge and save
print("\n" + "=" * 70)
if all_data:
    merged = pd.concat(all_data, ignore_index=True)
    
    # Clean the data
    merged = merged.dropna()
    merged['input'] = merged['input'].astype(str).str.strip()
    merged['response'] = merged['response'].astype(str).str.strip()
    merged = merged[merged['response'].str.len() > 0]
    merged = merged[merged['input'].str.len() > 0]
    
    # Remove duplicates
    merged = merged.drop_duplicates(subset=['input', 'response'])
    
    # Shuffle
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    output_file = 'data/merged_training_data.csv'
    merged.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"✓ SUCCESS! Saved to: {output_file}")
    print(f"✓ Total conversation pairs: {len(merged):,}")
    print("\nDataset breakdown:")
    print(merged['dataset'].value_counts().to_string())
    print("\nSample data (first 3 rows):")
    print(merged.head(3)[['input', 'response', 'dataset']].to_string())
else:
    print("✗ ERROR: No datasets were loaded successfully")
    print("Please check that files exist in data/ folder")

print("=" * 70)
