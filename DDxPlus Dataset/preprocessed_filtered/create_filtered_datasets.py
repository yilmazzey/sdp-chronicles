#!/usr/bin/env python3
"""
Create filtered datasets by creating one-hot encoded features and dropping raw columns
"""
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path

def parse_evidences(evidences_str):
    """Parse the EVIDENCES column"""
    try:
        parsed = json.loads(evidences_str.replace("'", '"'))
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return list(parsed.keys())
        return []
    except:
        return []

print("="*80)
print("CREATING FILTERED DATASETS")
print("="*80)

# Load datasets
base_path = "DDxPlus Dataset/preprocessed_stratified/"
train_df = pd.read_csv(f"{base_path}train_preprocessed.csv")
val_df = pd.read_csv(f"{base_path}validation_preprocessed.csv")
test_df = pd.read_csv(f"{base_path}test_preprocessed.csv")

print(f"\nLoaded datasets:")
print(f"  Train: {train_df.shape}")
print(f"  Validation: {val_df.shape}")
print(f"  Test: {test_df.shape}")

# 1. Create evidence features
print("\n1. Creating evidence features...")
all_evidence_keys = set()
for df in [train_df, val_df, test_df]:
    evidences_list = df['EVIDENCES'].apply(parse_evidences)
    for evidences in evidences_list:
        if isinstance(evidences, list):
            all_evidence_keys.update(evidences)

all_evidence_keys = sorted(list(all_evidence_keys))
print(f"  Found {len(all_evidence_keys)} unique evidence types")

def apply_evidence_features(df):
    """Apply one-hot encoding to evidences"""
    evidences_list = df['EVIDENCES'].apply(parse_evidences)
    evidence_features = {}
    for evidence_key in all_evidence_keys:
        def check_evidence(x):
            if isinstance(x, list):
                return 1 if evidence_key in x else 0
            return 0
        evidence_features[f'evidence_{evidence_key}'] = evidences_list.apply(check_evidence)
    return pd.DataFrame(evidence_features)

train_evidences = apply_evidence_features(train_df)
val_evidences = apply_evidence_features(val_df)
test_evidences = apply_evidence_features(test_df)

# Combine
train_df_extended = pd.concat([train_df.reset_index(drop=True), train_evidences.reset_index(drop=True)], axis=1)
val_df_extended = pd.concat([val_df.reset_index(drop=True), val_evidences.reset_index(drop=True)], axis=1)
test_df_extended = pd.concat([test_df.reset_index(drop=True), test_evidences.reset_index(drop=True)], axis=1)

print(f"  Created {len(all_evidence_keys)} evidence features")

# 2. Create initial evidence features
print("\n2. Creating initial evidence features...")
all_initial_evidences = set()
for df in [train_df_extended, val_df_extended, test_df_extended]:
    all_initial_evidences.update(df['INITIAL_EVIDENCE'].unique())

all_initial_evidences = sorted(list(all_initial_evidences))
print(f"  Found {len(all_initial_evidences)} unique initial evidences")

def encode_initial_evidence(df):
    features = pd.DataFrame(index=df.index)
    for evidence in all_initial_evidences:
        features[f'initial_{evidence}'] = (df['INITIAL_EVIDENCE'] == evidence).astype(int)
    return features

train_initial = encode_initial_evidence(train_df_extended)
val_initial = encode_initial_evidence(val_df_extended)
test_initial = encode_initial_evidence(test_df_extended)

# Combine
train_final = pd.concat([train_df_extended.reset_index(drop=True), train_initial.reset_index(drop=True)], axis=1)
val_final = pd.concat([val_df_extended.reset_index(drop=True), val_initial.reset_index(drop=True)], axis=1)
test_final = pd.concat([test_df_extended.reset_index(drop=True), test_initial.reset_index(drop=True)], axis=1)

print(f"  Created {len(all_initial_evidences)} initial evidence features")

# 3. Drop raw EVIDENCES and INITIAL_EVIDENCE columns
print("\n3. Dropping raw columns (EVIDENCES, INITIAL_EVIDENCE)...")
train_final = train_final.drop(columns=['EVIDENCES', 'INITIAL_EVIDENCE'], errors='ignore')
val_final = val_final.drop(columns=['EVIDENCES', 'INITIAL_EVIDENCE'], errors='ignore')
test_final = test_final.drop(columns=['EVIDENCES', 'INITIAL_EVIDENCE'], errors='ignore')

print(f"\n✓ Final dataset shapes:")
print(f"  Train: {train_final.shape}")
print(f"  Validation: {val_final.shape}")
print(f"  Test: {test_final.shape}")

# 4. Save datasets
output_path = Path("DDxPlus Dataset/preprocessed_filtered")
output_path.mkdir(exist_ok=True)

print(f"\n4. Saving filtered datasets to {output_path}...")
train_final.to_csv(output_path / "train_filtered.csv", index=False)
val_final.to_csv(output_path / "validation_filtered.csv", index=False)
test_final.to_csv(output_path / "test_filtered.csv", index=False)

# Save metadata
feature_cols = [col for col in train_final.columns if col not in ['PATHOLOGY', 'PATHOLOGY_ENCODED', 'DIFFERENTIAL_DIAGNOSIS', 'SEX']]

metadata = {
    'total_features': len(feature_cols),
    'evidence_features': len([c for c in feature_cols if c.startswith('evidence_')]),
    'initial_features': len([c for c in feature_cols if c.startswith('initial_')]),
    'dataset_shapes': {
        'train': train_final.shape,
        'validation': val_final.shape,
        'test': test_final.shape
    }
}

with open(output_path / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print("✓ Datasets saved successfully!")
print(f"\n✓ Summary:")
print(f"  - Total features: {len(feature_cols)}")
print(f"  - Evidence features: {metadata['evidence_features']}")
print(f"  - Initial evidence features: {metadata['initial_features']}")
print(f"  - Output: {output_path}/")

