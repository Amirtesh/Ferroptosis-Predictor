#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def create_output_dir():
    output_dir = Path("EDA_Results")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def load_and_prepare_data(file_path):
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    
    descriptor_cols = [col for col in df.columns if not col.startswith('MORGAN_bit_')]
    fingerprint_cols = [col for col in df.columns if col.startswith('MORGAN_bit_')]
    
    print(f"Molecular descriptors: {len(descriptor_cols) - 2}")
    print(f"Morgan fingerprint bits: {len(fingerprint_cols)}")
    
    numeric_cols = [col for col in descriptor_cols if col not in ['SMILES', 'Class']]
    
    nan_count = df[numeric_cols].isna().sum().sum()
    inf_count = np.isinf(df[numeric_cols].values).sum()
    
    print(f"\nCleaning data:")
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")
    
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    print(f"  After cleaning: NaN={df[numeric_cols].isna().sum().sum()}, Inf={np.isinf(df[numeric_cols].values).sum()}")
    
    return df, descriptor_cols, fingerprint_cols

def plot_class_distribution(df, output_dir):
    print("\n" + "="*70)
    print("1. CLASS DISTRIBUTION")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    class_counts = df['Class'].value_counts()
    colors = ['#e74c3c', '#3498db']
    
    axes[0].bar(class_counts.index, class_counts.values, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_xlabel('Class', fontsize=12)
    
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '01_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Classes: {class_counts.to_dict()}")
    print("✓ Saved: 01_class_distribution.png")

def plot_molecular_descriptors(df, descriptor_cols, output_dir):
    print("\n" + "="*70)
    print("2. MOLECULAR DESCRIPTORS ANALYSIS")
    print("="*70)
    
    all_descriptors = [col for col in descriptor_cols if col not in ['SMILES', 'Class']]
    
    colors = {'Ferroptosis-Inducer': '#e74c3c', 'Ferroptosis-Inhibitor': '#3498db'}
    
    descriptors_per_plot = 8
    num_plots = (len(all_descriptors) + descriptors_per_plot - 1) // descriptors_per_plot
    
    for plot_num in range(num_plots):
        start_idx = plot_num * descriptors_per_plot
        end_idx = min(start_idx + descriptors_per_plot, len(all_descriptors))
        current_descriptors = all_descriptors[start_idx:end_idx]
        
        fig, axes = plt.subplots(4, 2, figsize=(16, 16))
        axes = axes.flatten()
        
        for idx, desc in enumerate(current_descriptors):
            ax = axes[idx]
            for class_name in df['Class'].unique():
                data = df[df['Class'] == class_name][desc]
                ax.hist(data, bins=30, alpha=0.6, label=class_name,
                       color=colors[class_name], edgecolor='black')
            
            ax.set_title(f'{desc} Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel(desc, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        
        for idx in range(len(current_descriptors), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'02_molecular_descriptors_part{plot_num+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: 02_molecular_descriptors_part{plot_num+1}.png ({len(current_descriptors)} descriptors)")

def plot_descriptor_comparison_bars(df, descriptor_cols, output_dir):
    print("\n" + "="*70)
    print("3. INDUCER VS INHIBITOR COMPARISON")
    print("="*70)
    
    all_descriptors = [col for col in descriptor_cols if col not in ['SMILES', 'Class']]
    
    inducer_means = df[df['Class'] == 'Ferroptosis-Inducer'][all_descriptors].mean()
    inhibitor_means = df[df['Class'] == 'Ferroptosis-Inhibitor'][all_descriptors].mean()
    
    comparison_df = pd.DataFrame({
        'Inducer': inducer_means,
        'Inhibitor': inhibitor_means
    })
    
    descriptors_per_plot = 8
    num_plots = (len(all_descriptors) + descriptors_per_plot - 1) // descriptors_per_plot
    
    for plot_num in range(num_plots):
        start_idx = plot_num * descriptors_per_plot
        end_idx = min(start_idx + descriptors_per_plot, len(all_descriptors))
        current_descriptors = all_descriptors[start_idx:end_idx]
        
        fig, axes = plt.subplots(4, 2, figsize=(16, 16))
        axes = axes.flatten()
        
        for idx, desc in enumerate(current_descriptors):
            ax = axes[idx]
            x = ['Inducer', 'Inhibitor']
            y = [comparison_df.loc[desc, 'Inducer'], comparison_df.loc[desc, 'Inhibitor']]
            colors_bar = ['#e74c3c', '#3498db']
            
            bars = ax.bar(x, y, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
            ax.set_title(f'{desc}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Mean Value', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            
            for i, (bar, val) in enumerate(zip(bars, y)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        for idx in range(len(current_descriptors), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Mean Descriptor Values: Inducers vs Inhibitors (Part {plot_num+1})',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / f'03_inducer_vs_inhibitor_comparison_part{plot_num+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: 03_inducer_vs_inhibitor_comparison_part{plot_num+1}.png")

def plot_boxplots(df, descriptor_cols, output_dir):
    print("\n" + "="*70)
    print("4. BOXPLOT ANALYSIS")
    print("="*70)
    
    all_descriptors = [col for col in descriptor_cols if col not in ['SMILES', 'Class']]
    
    descriptors_per_plot = 8
    num_plots = (len(all_descriptors) + descriptors_per_plot - 1) // descriptors_per_plot
    
    for plot_num in range(num_plots):
        start_idx = plot_num * descriptors_per_plot
        end_idx = min(start_idx + descriptors_per_plot, len(all_descriptors))
        current_descriptors = all_descriptors[start_idx:end_idx]
        
        fig, axes = plt.subplots(4, 2, figsize=(16, 16))
        axes = axes.flatten()
        
        for idx, desc in enumerate(current_descriptors):
            ax = axes[idx]
            data_to_plot = [df[df['Class'] == 'Ferroptosis-Inducer'][desc],
                           df[df['Class'] == 'Ferroptosis-Inhibitor'][desc]]
            
            bp = ax.boxplot(data_to_plot, labels=['Inducer', 'Inhibitor'],
                           patch_artist=True, showmeans=True)
            
            for patch, color in zip(bp['boxes'], ['#e74c3c', '#3498db']):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax.set_title(f'{desc}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
        
        for idx in range(len(current_descriptors), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Descriptor Distributions: Boxplot Comparison (Part {plot_num+1})',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / f'04_boxplot_comparison_part{plot_num+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: 04_boxplot_comparison_part{plot_num+1}.png")

def plot_correlation_heatmap(df, descriptor_cols, output_dir):
    print("\n" + "="*70)
    print("5. CORRELATION ANALYSIS")
    print("="*70)
    
    numeric_descriptors = [col for col in descriptor_cols 
                          if col not in ['SMILES', 'Class']]
    
    corr_matrix = df[numeric_descriptors].corr()
    
    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
               square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
               vmin=-1, vmax=1, ax=ax)
    ax.set_title('Correlation Heatmap - All Descriptors', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '05_correlation_heatmap_full.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: 05_correlation_heatmap_full.png")
    
    key_descriptors = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds',
                       'NumHeavyAtoms', 'NumAromaticRings', 'FractionCsp3',
                       'MolMR', 'BertzCT', 'LabuteASA',
                       'EState_VSA2', 'VSA_EState2', 'EState_VSA1',
                       'SMR_VSA1', 'PEOE_VSA2', 'PEOE_VSA1',
                       'MinPartialCharge', 'MaxPartialCharge',
                       'Chi0v', 'Chi1v', 'BalabanJ', 'Kappa1']
    
    key_descriptors = [d for d in key_descriptors if d in df.columns]
    
    corr_matrix_key = df[key_descriptors].corr()
    
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr_matrix_key, annot=True, fmt='.2f', cmap='coolwarm',
               center=0, square=True, linewidths=1,
               cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1, ax=ax,
               annot_kws={'fontsize': 7})
    ax.set_title('Correlation Heatmap - Key Descriptors', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '06_correlation_heatmap_key.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: 06_correlation_heatmap_key.png")

def plot_fingerprint_analysis(df, fingerprint_cols, output_dir):
    print("\n" + "="*70)
    print("6. FINGERPRINT ANALYSIS")
    print("="*70)
    
    if len(fingerprint_cols) == 0:
        print("No fingerprint columns found")
        return
    
    fp_data = df[fingerprint_cols]
    
    sparsity = (fp_data == 0).sum().sum() / (fp_data.shape[0] * fp_data.shape[1]) * 100
    bits_per_molecule = fp_data.sum(axis=1)
    bit_frequency = fp_data.sum(axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    

    sparsity_text = f'Fingerprint Sparsity\n\n{sparsity:.2f}%\n\nof bits are zero'
    axes[0, 0].text(0.5, 0.5, sparsity_text,
                   ha='center', va='center', fontsize=18, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Overall Sparsity', fontsize=14, fontweight='bold', pad=20)
    

    for class_name in df['Class'].unique():
        class_bits = bits_per_molecule[df['Class'] == class_name]
        color = '#e74c3c' if 'Inducer' in class_name else '#3498db'
        axes[0, 1].hist(class_bits, bins=50, alpha=0.6, label=class_name,
                       color=color, edgecolor='black')
    
    axes[0, 1].set_xlabel('Number of Bits Set', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Distribution of Set Bits per Molecule', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].hist(bit_frequency, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Frequency (across molecules)', fontsize=12)
    axes[1, 0].set_ylabel('Number of Bits', fontsize=12)
    axes[1, 0].set_title('Bit Frequency Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    stats_text = f'Fingerprint Statistics:\n\n'
    stats_text += f'Total Bits: {len(fingerprint_cols)}\n'
    stats_text += f'Sparsity: {sparsity:.2f}%\n\n'
    stats_text += f'Bits per Molecule:\n'
    stats_text += f'  Mean: {bits_per_molecule.mean():.1f}\n'
    stats_text += f'  Median: {bits_per_molecule.median():.1f}\n'
    stats_text += f'  Std: {bits_per_molecule.std():.1f}\n\n'
    stats_text += f'Unique Bit Patterns:\n  {fp_data.drop_duplicates().shape[0]}'
    
    axes[1, 1].text(0.05, 0.5, stats_text, ha='left', va='center',
                   fontsize=10, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / '07_fingerprint_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sparsity: {sparsity:.2f}%")
    print(f"Mean bits set per molecule: {bits_per_molecule.mean():.1f}")
    print("✓ Saved: 07_fingerprint_analysis.png")

def plot_pairwise_relationships(df, output_dir):
    print("\n" + "="*70)
    print("7. PAIRWISE RELATIONSHIPS")
    print("="*70)
    
    key_vars = ['MW', 'LogP', 'TPSA', 'HBA', 'EState_VSA2']
    key_vars = [v for v in key_vars if v in df.columns]
    
    if len(key_vars) < 2:
        print("Not enough variables for pairwise plot")
        return
    
    plot_df = df[key_vars + ['Class']].copy()
    
    g = sns.pairplot(plot_df, hue='Class',
                    palette={'Ferroptosis-Inducer': '#e74c3c',
                            'Ferroptosis-Inhibitor': '#3498db'},
                    diag_kind='hist', plot_kws={'alpha': 0.6, 's': 30},
                    height=2.5)
    
    g.fig.suptitle('Pairwise Relationships - Key Descriptors',
                  fontsize=16, fontweight='bold', y=1.01)
    
    plt.savefig(output_dir / '08_pairwise_relationships.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: 08_pairwise_relationships.png")

def generate_summary_statistics(df, descriptor_cols, output_dir):
    print("\n" + "="*70)
    print("8. SUMMARY STATISTICS")
    print("="*70)
    
    numeric_descriptors = [col for col in descriptor_cols 
                          if col not in ['SMILES', 'Class']]
    
    overall_stats = df[numeric_descriptors].describe()
    overall_stats.to_csv(output_dir / 'summary_statistics_overall.csv')
    print("✓ Saved: summary_statistics_overall.csv")
    
    for class_name in df['Class'].unique():
        class_df = df[df['Class'] == class_name]
        class_stats = class_df[numeric_descriptors].describe()
        filename = class_name.replace('-', '_').replace(' ', '_').lower()
        class_stats.to_csv(output_dir / f'summary_statistics_{filename}.csv')
        print(f"✓ Saved: summary_statistics_{filename}.csv")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 eda.py <input_csv_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print("\n" + "="*70)
    print("FERROPTOSIS DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    output_dir = create_output_dir()
    print(f"\nOutput directory: {output_dir}")
    
    df, descriptor_cols, fingerprint_cols = load_and_prepare_data(file_path)
    
    plot_class_distribution(df, output_dir)
    plot_molecular_descriptors(df, descriptor_cols, output_dir)
    plot_descriptor_comparison_bars(df, descriptor_cols, output_dir)
    plot_boxplots(df, descriptor_cols, output_dir)
    plot_correlation_heatmap(df, descriptor_cols, output_dir)
    plot_fingerprint_analysis(df, fingerprint_cols, output_dir)
    plot_pairwise_relationships(df, output_dir)
    generate_summary_statistics(df, descriptor_cols, output_dir)
    
    print("\n" + "="*70)
    print("EDA COMPLETE!")
    print("="*70)
    print(f"All plots saved in: {output_dir}/")
    print("="*70)

if __name__ == "__main__":
    main()
