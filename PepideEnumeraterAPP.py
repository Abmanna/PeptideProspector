import itertools
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, Crippen, rdMolDescriptors, Fragments, rdFingerprintGenerator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# No sklearn, using numpy for split

try:
    import py3Dmol
    use_py3dmol = True
except ImportError:
    use_py3dmol = False

# Standard 20 amino acids with 1-letter codes
AMINO_ACIDS_FULL = {
    'A': 'Alanine', 'C': 'Cysteine', 'D': 'Aspartic Acid', 'E': 'Glutamic Acid',
    'F': 'Phenylalanine', 'G': 'Glycine', 'H': 'Histidine', 'I': 'Isoleucine',
    'K': 'Lysine', 'L': 'Leucine', 'M': 'Methionine', 'N': 'Asparagine',
    'P': 'Proline', 'Q': 'Glutamine', 'R': 'Arginine', 'S': 'Serine',
    'T': 'Threonine', 'V': 'Valine', 'W': 'Tryptophan', 'Y': 'Tyrosine'
}

# Enhanced amino acid property classifications
AMINO_ACID_PROPERTIES = {
    'hydrophobic': ['A', 'I', 'L', 'M', 'F', 'W', 'V', 'P'],
    'polar': ['S', 'T', 'N', 'Q', 'Y', 'C'],
    'charged_positive': ['K', 'R', 'H'],
    'charged_negative': ['D', 'E'],
    'charged': ['D', 'E', 'K', 'R', 'H'],
    'aromatic': ['F', 'W', 'Y', 'H'],
    'small': ['A', 'G', 'S', 'T', 'V'],
    'large': ['F', 'W', 'Y', 'R', 'K', 'L', 'I', 'M'],
    'flexible': ['G', 'S', 'T', 'A', 'D', 'N'],
    'rigid': ['P', 'W', 'F', 'Y'],
    'all': list(AMINO_ACIDS_FULL.keys()),
    
    # Target-specific amino acid sets
    'kinase_binding': ['F', 'W', 'Y', 'H', 'K', 'R', 'D', 'E', 'L', 'I', 'V'],  # ATP-binding site mimics
    'anticancer_active': ['R', 'K', 'F', 'W', 'L', 'I', 'V', 'C', 'H'],  # Cell-penetrating + cytotoxic
    'painkiller_active': ['F', 'W', 'Y', 'L', 'I', 'V', 'K', 'R', 'G'],  # Opioid receptor binding
    'membrane_penetrating': ['R', 'K', 'H', 'F', 'W', 'L', 'I', 'V'],  # Cell penetration
    'protease_resistant': ['D', 'E', 'P', 'G', 'C', 'W', 'F'],  # Protease stability
}

# Target-specific pharmacophore patterns (known active motifs)
PHARMACOPHORE_PATTERNS = {
    'kinase_inhibitor': {
        'required_patterns': [
            r'[FWY]',  # At least one aromatic for π-π stacking
            r'[KRH]',  # Positive charge for ATP site
        ],
        'favorable_patterns': [
            r'[FWY].*[KRH]',  # Aromatic followed by basic
            r'[LIVF][LIVF]',  # Hydrophobic clustering
            r'[FWY][LIVF]',   # Aromatic-hydrophobic
        ],
        'avoid_patterns': [
            r'PP',  # Proline-proline (rigid)
            r'[DE]{3,}',  # Too many negative charges
        ]
    },
    
    'anticancer': {
        'required_patterns': [
            r'[RK]',  # At least one positive for membrane interaction
            r'[FWLIV]',  # Hydrophobic for membrane insertion
        ],
        'favorable_patterns': [
            r'[RK].*[FWLIV]',  # Cationic amphiphilic
            r'[FWLIV]{2,}',    # Hydrophobic clustering
            r'[RK]{2,}',       # Multiple positive charges
            r'C.*C',           # Disulfide potential
        ],
        'avoid_patterns': [
            r'[DE]{2,}',  # Avoid negative charges
            r'GGG',       # Too flexible
        ]
    },
    
    'painkiller': {
        'required_patterns': [
            r'[FWY]',  # Aromatic for opioid receptor binding
            r'[KRHN]',  # Basic or polar for receptor interaction
        ],
        'favorable_patterns': [
            r'[FWY].*[KRH]',   # Aromatic-basic separation
            r'[FWY][FWY]',     # Multiple aromatics
            r'[LIVF][FWY]',    # Hydrophobic-aromatic
            r'[ST]',           # Serine/Threonine for H-bonding
        ],
        'avoid_patterns': [
            r'[DE]{2,}',  # Multiple negative charges
            r'PPP',       # Too rigid
        ]
    }
}

# Target-specific property filters
TARGET_FILTERS = {
    'kinase_inhibitor': {
        "MolWt": (200, 800),
        "LogP": (0, 4),
        "TPSA": (60, 140),
        "HBA": (3, 12),
        "HBD": (1, 6),
        "RotatableBonds": (2, 12),
        "NumAromaticRings": (1, 3),
        "FractionCsp3": (0.2, 0.8),
        "kinase_inhibitor_score": (0.5, 1.0)  # Custom scoring
    },
    
    'anticancer': {
        "MolWt": (300, 1200),
        "LogP": (1, 6),
        "TPSA": (40, 200),
        "HBA": (2, 15),
        "HBD": (0, 8),
        "RotatableBonds": (3, 20),
        "Charge": (1, 8),  # Prefer cationic
        "anticancer_score": (0.6, 1.0)
    },
    
    'painkiller': {
        "MolWt": (250, 900),
        "LogP": (1, 5),
        "TPSA": (50, 120),
        "HBA": (2, 10),
        "HBD": (1, 5),
        "RotatableBonds": (3, 15),
        "NumAromaticRings": (1, 3),
        "painkiller_score": (0.5, 1.0)
    },
    
    # General filters
    'drug_like': {
        "MolWt": (200, 500),
        "LogP": (-1, 3),
        "TPSA": (60, 120),
        "HBD": (0, 3),
        "HBA": (2, 8),
        "RotatableBonds": (2, 8)
    },
    
    'lipinski': {
        "MolWt": (0, 500),
        "LogP": (-3, 5),
        "HBD": (0, 5),
        "HBA": (0, 10)
    }
}

# Amino acid properties for scoring
AA_PROPERTIES = {
    'hydrophobicity': {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5,
                      'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
                      'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9,
                      'V': 4.2, 'Y': -1.3},
    
    'charge': {'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.5},
    'aromatic': {'F': 1, 'W': 1, 'Y': 1, 'H': 0.5},
    'polar': {'S': 1, 'T': 1, 'N': 1, 'Q': 1, 'Y': 1, 'C': 1}
}


# Hardcoded reference SMILES for targets (since no internet)
REFERENCE_SMILES = {
    'kinase_inhibitor': 'Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(c(c4)CN5CCN(CC5)C)C',  # Imatinib approximation
    'anticancer': 'CC(=O)OC1CC2CCC3C4CC(c5cc(OC)c(OC)c(OC)c5C4CCC3(C1OC(=O)C)C2O)C(O)C6OC(O)(C(=O)C(=O)N7CCCCC7C(=O)NC(c8ccccc8)c9ccc(O)c(c9)O)C(OC1C(O)C(O)C(O)C(C)O1)C6OC(=O)C',  # Doxorubicin, long
    'painkiller': 'CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O'  # Morphine
}


def calculate_target_score(sequence, target_type):
    """Calculate target-specific scoring based on sequence composition and patterns."""
    if target_type not in PHARMACOPHORE_PATTERNS:
        return 0.5
    
    patterns = PHARMACOPHORE_PATTERNS[target_type]
    score = 0.5  # Base score
    
    # Check required patterns
    required_met = sum(1 for pattern in patterns['required_patterns'] if re.search(pattern, sequence))
    
    if required_met == len(patterns['required_patterns']):
        score += 0.3
    else:
        score -= 0.2
    
    # Check favorable patterns
    favorable_count = sum(1 for pattern in patterns.get('favorable_patterns', []) if re.search(pattern, sequence))
    if len(patterns.get('favorable_patterns', [])) > 0:
        score += (favorable_count / len(patterns['favorable_patterns'])) * 0.3
    
    # Penalize avoid patterns
    for pattern in patterns.get('avoid_patterns', []):
        if re.search(pattern, sequence):
            score -= 0.2
    
    # Target-specific composition scoring
    if target_type == 'kinase_inhibitor':
        aromatic_frac = sum(1 for aa in sequence if aa in 'FWYH') / len(sequence)
        basic_frac = sum(1 for aa in sequence if aa in 'KRH') / len(sequence)
        score += (aromatic_frac + basic_frac) * 0.2
        
    elif target_type == 'anticancer':
        charge = sum(AA_PROPERTIES['charge'].get(aa, 0) for aa in sequence)
        hydrophobic_frac = sum(1 for aa in sequence if aa in 'FWLIVMA') / len(sequence)
        if charge > 0:
            score += 0.2
        score += hydrophobic_frac * 0.1
        
    elif target_type == 'painkiller':
        aromatic_frac = sum(1 for aa in sequence if aa in 'FWYH') / len(sequence)
        polar_frac = sum(1 for aa in sequence if aa in 'STNQYC') / len(sequence)
        score += aromatic_frac * 0.2 + polar_frac * 0.1
    
    return max(0, min(1, score))


def generate_peptide_sequences(length, amino_acids=None, exclude_patterns=None, target_type=None):
    """Enhanced sequence generation with target-specific biasing."""
    if amino_acids is None:
        key = target_type + '_active' if target_type and target_type + '_active' in AMINO_ACID_PROPERTIES else 'all'
        amino_acids = AMINO_ACID_PROPERTIES[key]
    
    for seq in itertools.product(amino_acids, repeat=length):
        seq_str = ''.join(seq)
        
        if exclude_patterns and any(re.search(pattern, seq_str) for pattern in exclude_patterns):
            continue
        
        if target_type:
            score = calculate_target_score(seq_str, target_type)
            if score < 0.3:
                continue
                
        yield seq_str


def build_optimized_peptide(sequence, n_confs=10, max_attempts=3, force_field='MMFF', cyclic=False):
    """Enhanced peptide building with optional cyclization."""
    mol = Chem.MolFromSequence(sequence)
    if mol is None:
        raise ValueError(f"Invalid sequence: {sequence}")

    if cyclic:
        # Simple cyclization, may need better implementation
        smiles = '[N:1]' + sequence.replace(' ', '') + '[C:2](=O)'
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Cyclization failed")
    
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.numThreads = 0
    params.useSmallRingTorsions = True
    params.randomSeed = 42

    for attempt in range(max_attempts):
        try:
            conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
            if len(conf_ids) == 0:
                continue

            energies = []
            for conf_id in conf_ids:
                try:
                    if force_field.upper() == 'MMFF':
                        ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf_id)
                    else:
                        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                    
                    if ff is None:
                        continue
                        
                    ff.Minimize(maxIts=500)
                    energy = ff.CalcEnergy()
                    energies.append((energy, conf_id))
                except:
                    continue

            if not energies:
                continue

            energies.sort()
            best_conf = energies[0][1]
            return mol, best_conf, energies[0][0]

        except Exception as e:
            if attempt == max_attempts - 1:
                raise RuntimeError(f"Failed to build {sequence}: {e}")
    
    raise RuntimeError(f"Conformer generation failed for {sequence}")


def compute_comprehensive_properties(mol, sequence, target_type=None):
    """Comprehensive property calculation including target-specific descriptors and fingerprints."""
    props = {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "RotatableBonds": Descriptors.NumRotatableBonds(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "Charge": Chem.GetFormalCharge(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),
        "FractionCsp3": rdMolDescriptors.CalcFractionCsp3(mol),
        "MolMR": Crippen.MolMR(mol),
        "BalabanJ": rdMolDescriptors.BalabanJ(mol),
        "BertzCT": rdMolDescriptors.BertzCT(mol),
        "NumAliphCarbocycles": Fragments.fr_Al_COO(mol),
        "NumArylCarbocycles": Fragments.fr_Ar_N(mol),
        "NumBasicGroups": Fragments.fr_NH2(mol) + Fragments.fr_NH1(mol),
    }
    
    # Fingerprint
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    props['fingerprint'] = fp_gen.GetFingerprint(mol)
    
    composition = Counter(sequence)
    total_aa = len(sequence)
    
    for prop_name, aa_list in AMINO_ACID_PROPERTIES.items():
        if prop_name != 'all':
            count = sum(composition.get(aa, 0) for aa in aa_list)
            props[f'frac_{prop_name}'] = count / total_aa if total_aa > 0 else 0
    
    props['avg_hydrophobicity'] = np.mean([AA_PROPERTIES['hydrophobicity'].get(aa, 0) for aa in sequence])
    props['net_charge'] = sum(AA_PROPERTIES['charge'].get(aa, 0) for aa in sequence)
    props['aromatic_content'] = sum(AA_PROPERTIES['aromatic'].get(aa, 0) for aa in sequence)
    props['polar_content'] = sum(AA_PROPERTIES['polar'].get(aa, 0) for aa in sequence)
    
    if target_type:
        props[f'{target_type}_score'] = calculate_target_score(sequence, target_type)
    
    props['lipinski_violations'] = sum([
        props['MolWt'] > 500,
        props['LogP'] > 5,
        props['HBD'] > 5,
        props['HBA'] > 10
    ])
    
    props['oral_bioavailability'] = 1.0 if (
        props['MolWt'] <= 500 and props['TPSA'] <= 140 and
        props['RotatableBonds'] <= 10 and props['lipinski_violations'] <= 1
    ) else 0.0
    
    return props


def calculate_similarity(mol_fp, ref_fp):
    if ref_fp is None:
        return 0.0
    return Chem.DataStructs.TanimotoSimilarity(mol_fp, ref_fp)


def get_reference_fp(target_type):
    if target_type in REFERENCE_SMILES:
        mol = Chem.MolFromSmiles(REFERENCE_SMILES[target_type])
        if mol:
            fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
            return fp_gen.GetFingerprint(mol)
    return None


def analyze_target_specificity(df, target_type):
    if target_type not in PHARMACOPHORE_PATTERNS:
        return {}
    
    patterns = PHARMACOPHORE_PATTERNS[target_type]
    analysis = {}
    
    for pattern_type, pattern_list in patterns.items():
        analysis[pattern_type] = {}
        for i, pattern in enumerate(pattern_list):
            matches = df['Sequence'].str.contains(pattern, regex=True).sum()
            analysis[pattern_type][f'pattern_{i+1}'] = {
                'pattern': pattern,
                'matches': matches,
                'frequency': matches / len(df) if len(df) > 0 else 0
            }
    
    target_aa = AMINO_ACID_PROPERTIES.get(target_type + '_active', [])
    if target_aa:
        analysis['target_aa_usage'] = {}
        for aa in target_aa:
            usage = df['Sequence'].str.count(aa).sum()
            analysis['target_aa_usage'][aa] = usage
    
    return analysis


def create_target_specific_plots(df, target_type, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8')
    
    score_col = f'{target_type}_score'
    if score_col in df.columns and len(df) > 0:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[score_col], bins=20, kde=True)
        plt.axvline(df[score_col].mean(), color='red', linestyle='--', label=f'Mean: {df[score_col].mean():.3f}')
        plt.xlabel(f'{target_type.title()} Score')
        plt.ylabel('Frequency')
        plt.title(f'{target_type.title()} Score Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{target_type}_score_dist.png", dpi=300)
        plt.close()
    
    if score_col in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        properties = ['MolWt', 'LogP', 'TPSA', 'net_charge']
        
        for i, prop in enumerate(properties):
            if prop in df.columns:
                ax = axes[i//2, i%2]
                sns.scatterplot(x=df[prop], y=df[score_col], hue=df['Length'], ax=ax, palette='viridis', alpha=0.6)
                ax.set_title(f'{prop} vs {target_type.title()} Score')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{target_type}_property_correlations.png", dpi=300)
        plt.close()
    
    aa_columns = [col for col in df.columns if col.startswith('frac_') and col.replace('frac_', '') in AMINO_ACIDS_FULL]
    
    if aa_columns:
        plt.figure(figsize=(12, 8))
        aa_data = df[aa_columns].mean().to_frame().T
        aa_names = [col.replace('frac_', '') for col in aa_columns]
        sns.heatmap(aa_data, annot=True, fmt='.3f', xticklabels=aa_names, yticklabels=['Average Fraction'], cmap='YlOrRd')
        plt.title(f'Amino Acid Usage in {target_type.title()} Peptides')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{target_type}_aa_usage.png", dpi=300)
        plt.close()


class PeptideDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class PeptideClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


def train_ml_classifier(df, target_type, epochs=50):
    """Train a simple ML classifier for target suitability using Torch."""
    score_col = f'{target_type}_score'
    if score_col not in df.columns or len(df) < 20:  # Increase min size
        print("Insufficient data for ML training")
        return None, None
    
    features = ['MolWt', 'LogP', 'TPSA', 'net_charge', 'aromatic_content', 'avg_hydrophobicity']
    X = df[features].to_numpy()
    y = (df[score_col] > df[score_col].median()).astype(float)  # Binary: above median
    
    # Manual split with numpy
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(0.8 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    train_ds = PeptideDataset(X_train, y_train)
    val_ds = PeptideDataset(X_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    
    model = PeptideClassifier(len(features))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        val_losses = []
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
    
    return model, val_loss


def predict_with_model(model, props):
    features = [props.get(key, 0) for key in ['MolWt', 'LogP', 'TPSA', 'net_charge', 'aromatic_content', 'avg_hydrophobicity']]
    input_tensor = torch.tensor([features], dtype=torch.float32)
    return model(input_tensor).item()


def run_specialized_peptide_discovery(
    target_type='kinase_inhibitor',
    min_length=3,
    max_length=6,
    amino_acids=None,
    custom_filters=None,
    exclude_patterns=None,
    n_confs=8,
    max_peptides=500,
    min_target_score=0.4,
    output_csv=None,
    create_plots=True,
    show_detailed_analysis=True,
    save_structures=False,
    use_ml=False,
    cyclic=False,
    use_similarity=False
):
    if output_csv is None:
        output_csv = f"{target_type}_peptides.csv"
    
    if amino_acids is None:
        amino_acids = AMINO_ACID_PROPERTIES.get(target_type + '_active', AMINO_ACID_PROPERTIES['all'])
    
    if custom_filters is None:
        custom_filters = TARGET_FILTERS.get(target_type, TARGET_FILTERS['drug_like'])
    
    if exclude_patterns is None:
        exclude_patterns = PHARMACOPHORE_PATTERNS.get(target_type, {}).get('avoid_patterns', [])
    
    print(f"🎯 SPECIALIZED PEPTIDE DISCOVERY: {target_type.upper()}")
    print(f"📏 Length range: {min_length}–{max_length}")
    print(f"🧬 Amino acid set: {sorted(set(amino_acids))}")
    print(f"🎯 Min target score: {min_target_score}")
    print(f"🚫 Excluded patterns: {exclude_patterns}")
    print(f"📊 Max evaluations: {max_peptides}")
    if use_ml:
        print("🤖 Using ML classification")
    if cyclic:
        print("🔄 Cyclic peptides enabled")
    if use_similarity:
        print("📏 Using structure-based similarity screening")
    
    results = []
    failed_sequences = []
    low_score_skipped = 0
    count = 0
    
    ref_fp = get_reference_fp(target_type) if use_similarity else None
    if use_similarity and ref_fp:
        custom_filters['similarity'] = (0.3, 1.0)
    
    ml_model = None

    for length in range(min_length, max_length + 1):
        print(f"\n🔄 Processing {length}-mer peptides...")
        length_count = 0
        for seq in generate_peptide_sequences(length, amino_acids, exclude_patterns, target_type):
            if count >= max_peptides:
                break

            target_score = calculate_target_score(seq, target_type)
            if target_score < min_target_score:
                low_score_skipped += 1
                continue

            try:
                mol, best_conf, energy = build_optimized_peptide(seq, n_confs=n_confs, cyclic=cyclic)
                props = compute_comprehensive_properties(mol, seq, target_type)
                props['Sequence'] = seq
                props['Length'] = length
                props['Energy'] = energy
                
                if ref_fp:
                    props['similarity'] = calculate_similarity(props['fingerprint'], ref_fp)
                
                if ml_model:
                    props['ml_score'] = predict_with_model(ml_model, props)
                
                if passes_filters(props, custom_filters):
                    results.append(props)
                    length_count += 1
                    print(f"✅ {seq} (Score={target_score:.3f}, MW={props['MolWt']:.1f})")
                    
                    if save_structures:
                        os.makedirs(f"structures_{target_type}", exist_ok=True)
                        with Chem.SDWriter(f"structures_{target_type}/{seq}.sdf") as writer:
                            writer.write(mol, confId=best_conf)

                count += 1

            except Exception as e:
                failed_sequences.append((seq, str(e)))
                print(f"❌ Failed {seq}: {e}")

        print(f"✅ Found {length_count} qualifying {length}-mers")
        if count >= max_peptides:
            break

    if results:
        df = pd.DataFrame(results)
        score_col = f'{target_type}_score'
        sort_cols = [score_col, 'MolWt', 'Energy'] if score_col in df.columns else ['MolWt', 'Energy']
        df = df.sort_values(sort_cols, ascending=[False, True, True])
        
        df.to_csv(output_csv, index=False)
        
        if create_plots and len(df) > 5:
            create_target_specific_plots(df, target_type)
        
        if show_detailed_analysis:
            analysis = analyze_target_specificity(df, target_type)
            print("\n📊 TARGET-SPECIFIC ANALYSIS:")
            if 'required_patterns' in analysis:
                print("Required patterns compliance:")
                for info in analysis['required_patterns'].values():
                    print(f"  {info['pattern']}: {info['frequency']:.1%} coverage")
        
        if use_ml:
            ml_model, val_loss = train_ml_classifier(df, target_type)
            if ml_model:
                print(f"\n🤖 ML Model trained with validation loss: {val_loss:.4f}")
                df['ml_score'] = df.apply(lambda row: predict_with_model(ml_model, row), axis=1)
                df = df.sort_values('ml_score', ascending=False)
                df.to_csv(output_csv, index=False)
        
        print(f"\n🎉 DISCOVERY COMPLETE!")
        print(f"✅ Found {len(df)} peptides")
        print(f"⏭️ Skipped {low_score_skipped} low scores")
        print(f"❌ {len(failed_sequences)} failed")
        print(f"📊 Saved to {output_csv}")
        
        display_cols = ["Sequence", "Length", score_col, "MolWt", "LogP", "TPSA", "net_charge", "aromatic_content"]
        available = [col for col in display_cols if col in df.columns]
        print("\n🏆 TOP 15 CANDIDATES:")
        print(df[available].head(15).to_string(index=False, float_format='%.3f'))
        
        if score_col in df.columns:
            print(f"\n📈 SCORE STATS:")
            print(f"Mean: {df[score_col].mean():.3f}")
            print(f"Best: {df[score_col].max():.3f}")
            print(f"Std: {df[score_col].std():.3f}")
        
        print("\n🧬 COMPOSITION SUMMARY:")
        print(f"Lengths: {dict(df['Length'].value_counts().sort_index())}")
        print(f"MW range: {df['MolWt'].min():.1f} - {df['MolWt'].max():.1f}")
        print(f"LogP range: {df['LogP'].min():.2f} - {df['LogP'].max():.2f}")
        if 'net_charge' in df.columns:
            print(f"Charge range: {df['net_charge'].min()} - {df['net_charge'].max()}")
        
        return df
    else:
        print("❌ No peptides found")
        return pd.DataFrame()


def compare_targets(output_dir="comparison_results"):
    os.makedirs(output_dir, exist_ok=True)
    targets = ['kinase_inhibitor', 'anticancer', 'painkiller']
    results = {}
    
    for target in targets:
        df = run_specialized_peptide_discovery(
            target_type=target,
            min_length=3,
            max_length=5,
            max_peptides=200,
            create_plots=False,
            show_detailed_analysis=False
        )
        results[target] = df
    
    if all(not df.empty for df in results.values()):
        plt.style.use('seaborn-v0_8')
        
        # Score comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for i, (target, df) in enumerate(results.items()):
            score_col = f'{target}_score'
            if score_col in df.columns:
                sns.histplot(df[score_col], ax=axes[i], bins=15, kde=True)
                axes[i].set_title(f'{target.title()} Scores')
                axes[i].set_xlabel('Target Score')
                axes[i].set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/target_score_comparison.png", dpi=300)
        plt.close()
        
        # Property comparison
        properties = ['MolWt', 'LogP', 'TPSA', 'net_charge']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        for i, prop in enumerate(properties):
            ax = axes[i//2, i%2]
            for target, df in results.items():
                if prop in df.columns:
                    sns.histplot(df[prop], ax=ax, alpha=0.5, label=target.title(), kde=True)
            ax.set_title(f'{prop} Distribution by Target')
            ax.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/property_comparison.png", dpi=300)
        plt.close()
        
        print("\n📊 TARGET COMPARISON SUMMARY:")
        for target, df in results.items():
            score_col = f'{target}_score'
            print(f"\n{target.upper()}:")
            print(f"  Peptides found: {len(df)}")
            if score_col in df.columns:
                print(f"  Avg score: {df[score_col].mean():.3f}")
                print(f"  Best score: {df[score_col].max():.3f}")
            print(f"  MW range: {df['MolWt'].min():.0f}-{df['MolWt'].max():.0f}")
            print(f"  LogP range: {df['LogP'].min():.2f}-{df['LogP'].max():.2f}")
    
    return results


def generate_pharmacophore_report(sequence, target_type):
    if target_type not in PHARMACOPHORE_PATTERNS:
        return "No pharmacophore data available for this target type."
    
    patterns = PHARMACOPHORE_PATTERNS[target_type]
    report = []
    
    report.append(f"PHARMACOPHORE ANALYSIS FOR {sequence} ({target_type.upper()})")
    report.append("=" * 60)
    
    # Required patterns
    report.append("\n🔴 REQUIRED PATTERNS:")
    for pattern in patterns['required_patterns']:
        matches = bool(re.search(pattern, sequence))
        status = "✅ PASS" if matches else "❌ FAIL"
        report.append(f"  {pattern}: {status}")
    
    # Favorable patterns
    report.append("\n🟡 FAVORABLE PATTERNS:")
    for pattern in patterns.get('favorable_patterns', []):
        matches = bool(re.search(pattern, sequence))
        status = "✅ PRESENT" if matches else "⚪ ABSENT"
        report.append(f"  {pattern}: {status}")
    
    # Avoid patterns
    report.append("\n🔴 PATTERNS TO AVOID:")
    for pattern in patterns.get('avoid_patterns', []):
        matches = bool(re.search(pattern, sequence))
        status = "❌ PRESENT" if matches else "✅ CLEAR"
        report.append(f"  {pattern}: {status}")
    
    # Composition analysis
    report.append(f"\n🧬 COMPOSITION ANALYSIS:")
    composition = Counter(sequence)
    total_aa = len(sequence)
    
    target_aa = AMINO_ACID_PROPERTIES.get(target_type + '_active', [])
    if target_aa:
        target_content = sum(composition.get(aa, 0) for aa in target_aa) / total_aa
        report.append(f"  Target-relevant AA content: {target_content:.1%}")
    
    for category, aa_list in AMINO_ACID_PROPERTIES.items():
        if category not in ['all', target_type + '_active']:
            content = sum(composition.get(aa, 0) for aa in aa_list) / total_aa
            if content > 0:
                report.append(f"  {category.title()} content: {content:.1%}")
    
    # Overall score
    score = calculate_target_score(sequence, target_type)
    report.append(f"\n🎯 OVERALL {target_type.upper()} SCORE: {score:.3f}")
    
    return "\n".join(report)


# Gradio Interface (run this in an environment where Gradio is installed)
import gradio as gr

def gradio_discovery(target_type, min_length, max_length, max_peptides, min_score, use_ml, cyclic, use_similarity):
    df = run_specialized_peptide_discovery(
        target_type=target_type,
        min_length=int(min_length),
        max_length=int(max_length),
        max_peptides=int(max_peptides),
        min_target_score=float(min_score),
        create_plots=True,
        show_detailed_analysis=True,
        use_ml=use_ml,
        cyclic=cyclic,
        use_similarity=use_similarity
    )
    
    if df.empty:
        return "No peptides found", None, None
    
    output_csv = f"{target_type}_peptides.csv"
    
    top_seq = df.iloc[0]['Sequence']
    report = generate_pharmacophore_report(top_seq, target_type)
    
    return report, df.to_html(index=False), output_csv

with gr.Blocks() as demo:
    gr.Markdown("# Peptide-Oriented Drug Design App")
    
    with gr.Row():
        target = gr.Dropdown(["kinase_inhibitor", "anticancer", "painkiller"], label="Target Type")
        min_len = gr.Number(value=3, label="Min Peptide Length")
        max_len = gr.Number(value=6, label="Max Peptide Length")
    
    with gr.Row():
        max_pep = gr.Number(value=500, label="Max Peptides to Generate")
        min_sc = gr.Number(value=0.4, label="Min Target Score")
    
    with gr.Row():
        use_ml_chk = gr.Checkbox(label="Use ML Classification")
        cyclic_chk = gr.Checkbox(label="Enable Cyclic Peptides")
        sim_chk = gr.Checkbox(label="Use Structure Similarity Screening")
    
    submit_btn = gr.Button("Run Discovery")
    
    report_output = gr.Textbox(label="Pharmacophore Report for Top Candidate", lines=20)
    table_output = gr.HTML(label="Discovery Results")
    csv_download = gr.File(label="Download Results CSV")
    
    submit_btn.click(
        gradio_discovery,
        inputs=[target, min_len, max_len, max_pep, min_sc, use_ml_chk, cyclic_chk, sim_chk],
        outputs=[report_output, table_output, csv_download]
    )

if __name__ == "__main__":
    demo.launch()