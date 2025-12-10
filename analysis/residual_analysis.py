"""
Probability Residual Analysis: Comparing Model Predictions vs Human Consensus
Analyzes the systematic biases between AI predictions and 400-rater human consensus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Setup
DATA_DIR = "./data/Taiwanese"
PREDICTIONS_CSV = "./data/Taiwanese/predictions_20251211_013516.csv"
VOTING_CSV = "./data/Taiwanese/taiwanese_ground_truth.csv"

EMOTION_CLASSES = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
EMOTION_COLS = ['angry', 'disgust', 'fear', 'happy', 'peace', 'sad', 'surprise']

def load_data():
    """Load predictions and voting data"""
    predictions_df = pd.read_csv(PREDICTIONS_CSV)
    voting_df = pd.read_csv(VOTING_CSV)
    return predictions_df, voting_df

def normalize_voting_probabilities(voting_df):
    """
    L1-Normalize voting probabilities (Softmax-compatible normalization)
    
    Rationale:
    Since raw voting data may sum > 1.0 (avg 1.107) due to multi-label voting,
    we normalize using L1 normalization to convert "absolute vote counts" to 
    "relative emotion saliency distribution":
    
    P_norm(c) = V(c) / sum(V(i))
    
    This ensures human perception data and model output probabilities 
    are on the same mathematical scale for fair comparison.
    """
    voting_cols = EMOTION_COLS
    
    # Calculate sum per sample
    vote_sum = voting_df[voting_cols].sum(axis=1)
    
    # Check normalization needed
    print("="*70)
    print("VOTING DATA NORMALIZATION ANALYSIS")
    print("="*70)
    print(f"\nOriginal voting probability statistics:")
    print(f"  Mean sum: {vote_sum.mean():.4f}")
    print(f"  Min sum: {vote_sum.min():.4f}")
    print(f"  Max sum: {vote_sum.max():.4f}")
    print(f"  Std dev: {vote_sum.std():.4f}")
    
    # Apply L1 normalization
    normalized = voting_df[voting_cols].div(vote_sum, axis=0)
    
    print(f"\nAfter L1-Normalization:")
    normalized_sum = normalized.sum(axis=1)
    print(f"  Mean sum: {normalized_sum.mean():.4f}")
    print(f"  Min sum: {normalized_sum.min():.4f}")
    print(f"  Max sum: {normalized_sum.max():.4f}")
    
    return normalized

def compute_residuals(predictions_df, voting_normalized):
    """
    Compute probability residuals: R = P_model - P_human
    
    For each sample and emotion class:
    - Positive residual: Model over-confident (predicts higher than human)
    - Negative residual: Model under-confident (predicts lower than human)
    - Near-zero: Perfect alignment
    """
    residuals = {}
    
    print(f"\n{'='*70}")
    print("COMPUTING PROBABILITY RESIDUALS")
    print("="*70)
    
    for idx, emotion in enumerate(EMOTION_CLASSES):
        emotion_col_pred = f'prob_{emotion}'
        emotion_col_voting = EMOTION_COLS[idx]
        
        # Extract probabilities
        model_probs = predictions_df[emotion_col_pred].values
        human_probs = voting_normalized[emotion_col_voting].values
        
        # Compute residuals
        residual = model_probs - human_probs
        residuals[emotion] = residual
        
        print(f"\n{emotion}:")
        print(f"  Model mean prob: {model_probs.mean():.4f}")
        print(f"  Human mean prob: {human_probs.mean():.4f}")
        print(f"  Residual mean: {residual.mean():.4f} (Model - Human)")
        print(f"  Residual std: {residual.std():.4f}")
        print(f"  Residual range: [{residual.min():.4f}, {residual.max():.4f}]")
    
    return residuals

def analyze_residuals(residuals):
    """
    Statistical analysis of residuals
    Identifies systematic biases for each emotion
    """
    print(f"\n{'='*70}")
    print("RESIDUAL BIAS ANALYSIS (t-tests)")
    print("="*70)
    print("\nTesting if mean residual is significantly different from 0")
    print("(Null hypothesis: No systematic bias)\n")
    
    bias_summary = {}
    
    for emotion, residual in residuals.items():
        t_stat, p_value = stats.ttest_1samp(residual, 0)
        mean_residual = residual.mean()
        
        bias_summary[emotion] = {
            'mean_residual': mean_residual,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        bias_direction = "OVER-CONFIDENT" if mean_residual > 0 else "UNDER-CONFIDENT"
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        
        print(f"{emotion:12} | Mean Residual: {mean_residual:+.4f} | {bias_direction:18} | p={p_value:.4f} {significance}")
    
    return bias_summary

def plot_residual_boxplots(residuals):
    """
    Create comprehensive boxplot visualization of residuals
    Shows distribution for each emotion class
    """
    print(f"\n{'='*70}")
    print("GENERATING BOXPLOT VISUALIZATION")
    print("="*70)
    
    # Prepare data for boxplot
    residual_data = []
    emotion_labels = []
    
    for emotion, residual in residuals.items():
        residual_data.extend(residual)
        emotion_labels.extend([emotion] * len(residual))
    
    plot_df = pd.DataFrame({
        'Residual': residual_data,
        'Emotion': emotion_labels
    })
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Boxplot
    ax1 = axes[0]
    sns.boxplot(data=plot_df, x='Emotion', y='Residual', ax=ax1, palette='Set2')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Perfect Alignment (R=0)')
    ax1.set_ylabel('Residual (P_model - P_human)', fontsize=12)
    ax1.set_xlabel('Emotion Class', fontsize=12)
    ax1.set_title('Probability Residual Distribution by Emotion\n(Positive=Over-confident, Negative=Under-confident)', 
                   fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Violin plot for distribution comparison
    ax2 = axes[1]
    sns.violinplot(data=plot_df, x='Emotion', y='Residual', ax=ax2, palette='Set2')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Perfect Alignment (R=0)')
    ax2.set_ylabel('Residual (P_model - P_human)', fontsize=12)
    ax2.set_xlabel('Emotion Class', fontsize=12)
    ax2.set_title('Residual Distribution (Violin Plot)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('./residual_boxplot.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: residual_boxplot.png")
    plt.close()

def plot_correlation_analysis(predictions_df, voting_normalized):
    """
    Correlation analysis: Human score vs Model score
    Shows if model learned to rank emotions correctly (trend detection)
    
    High correlation (r > 0.8): Model learned this emotion well
    Low correlation (r < 0.4): Model is essentially guessing
    """
    print(f"\nGenerating correlation analysis plots...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    correlations = {}
    
    for idx, emotion in enumerate(EMOTION_CLASSES):
        ax = axes[idx]
        
        emotion_col_pred = f'prob_{emotion}'
        emotion_col_voting = EMOTION_COLS[idx]
        
        human_scores = voting_normalized[emotion_col_voting].values
        model_scores = predictions_df[emotion_col_pred].values
        
        # Compute Pearson correlation
        correlation, p_value = stats.pearsonr(human_scores, model_scores)
        correlations[emotion] = {
            'r': correlation,
            'p_value': p_value
        }
        
        # Scatter plot with trend line
        ax.scatter(human_scores, model_scores, alpha=0.4, s=25, edgecolors='black', linewidth=0.5)
        
        # Add regression line
        z = np.polyfit(human_scores, model_scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(human_scores.min(), human_scores.max(), 100)
        ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f'Trend line')
        
        # Perfect correlation line (y=x)
        max_val = max(human_scores.max(), model_scores.max())
        ax.plot([0, max_val], [0, max_val], 'g--', linewidth=1.5, alpha=0.5, label='Perfect correlation')
        
        ax.set_xlabel('Human Score', fontsize=10)
        ax.set_ylabel('Model Score', fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        
        # Color code based on correlation strength
        if correlation > 0.8:
            title_color = 'green'
            strength = 'STRONG'
        elif correlation > 0.6:
            title_color = 'orange'
            strength = 'MODERATE'
        elif correlation > 0.4:
            title_color = 'red'
            strength = 'WEAK'
        else:
            title_color = 'darkred'
            strength = 'POOR'
        
        ax.set_title(f'{emotion}\nr={correlation:.3f} {strength}', 
                    fontsize=11, fontweight='bold', color=title_color)
        ax.grid(True, alpha=0.3)
        
        # Add statistics box
        stats_text = f'r = {correlation:.3f}\np < 0.001' if p_value < 0.001 else f'r = {correlation:.3f}\np = {p_value:.4f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                fontsize=9)
    
    axes[-1].remove()  # Remove last empty subplot
    
    plt.suptitle('Correlation Analysis: Does Model Learn Emotion Ranking?\n(High r = Model learned the emotion; Low r = Model guessing)', 
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('./correlation_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: correlation_analysis.png")
    plt.close()
    
    return correlations

def generate_summary_report(predictions_df, voting_normalized, residuals, bias_summary, correlations):
    """Generate comprehensive text report"""
    
    report = f"""
{'='*80}
PROBABILITY RESIDUAL ANALYSIS: AI vs Human Consensus
{'='*80}

METHODOLOGY:
-----------
Residual Formula: R_emotion = P_model(emotion) - P_human(emotion)

Interpretation:
  R > 0 (Positive):  Model OVER-CONFIDENT (predicts higher than human)
  R < 0 (Negative):  Model UNDER-CONFIDENT (predicts lower than human)
  R ≈ 0 (Near-zero): PERFECT ALIGNMENT

Data Processing:
  - Ground truth probabilities normalized using L1-normalization
  - Converts multi-label voting (sum ≤ 1.107) to Softmax-compatible distribution
  - P_norm(c) = V(c) / sum(V(i))
  - Ensures fair comparison between human and model probability spaces

{'='*80}
KEY FINDINGS
{'='*80}

1. SYSTEMATIC BIAS PATTERNS BY EMOTION:
"""
    
    for emotion in EMOTION_CLASSES:
        bias = bias_summary[emotion]
        direction = "OVER" if bias['mean_residual'] > 0 else "UNDER"
        significance = "***" if bias['p_value'] < 0.001 else "**" if bias['p_value'] < 0.01 else "*" if bias['p_value'] < 0.05 else "(ns)"
        
        report += f"\n  {emotion:12} | Mean R: {bias['mean_residual']:+.4f} | {direction}-confident {significance}"
    
    # Identify problem emotions
    report += f"\n\n2. PROBLEM EMOTIONS (Systematic Misalignment):\n"
    
    for emotion, residual in residuals.items():
        # Check if consistently under/over-confident
        positive_count = (residual > 0).sum()
        negative_count = (residual < 0).sum()
        total = len(residual)
        
        if positive_count > total * 0.7:
            report += f"  ⚠️  {emotion:12} | {positive_count}/{total} samples over-confident ({positive_count/total*100:.1f}%)\n"
        elif negative_count > total * 0.7:
            report += f"  ⚠️  {emotion:12} | {negative_count}/{total} samples under-confident ({negative_count/total*100:.1f}%)\n"
    
    # Identify unstable emotions
    report += f"\n3. UNSTABLE EMOTIONS (High Residual Variance):\n"
    residual_stds = {emotion: residual.std() for emotion, residual in residuals.items()}
    sorted_stds = sorted(residual_stds.items(), key=lambda x: x[1], reverse=True)
    
    for emotion, std in sorted_stds[:3]:
        report += f"  ⚠️  {emotion:12} | Std Dev: {std:.4f} (unstable predictions)\n"
    
    # Add correlation analysis
    report += f"""

4. CORRELATION ANALYSIS: Does Model Learn Emotion Ranking?

Interpretation:
  r > 0.8 (STRONG):   Model learned this emotion well (understands ranking)
  r > 0.6 (MODERATE): Reasonable correlation (partial understanding)
  r < 0.4 (WEAK):     Model barely correlates with human scores (guessing)
  
Emotion Rankings:
"""
    
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1]['r'], reverse=True)
    for emotion, corr_data in sorted_corr:
        r = corr_data['r']
        if r > 0.8:
            strength = "✓✓✓ STRONG - Model understands this emotion"
        elif r > 0.6:
            strength = "✓✓ MODERATE - Partial understanding"
        elif r > 0.4:
            strength = "✓ WEAK - Limited understanding"
        else:
            strength = "✗ POOR - Model essentially guessing"
        
        report += f"\n  {emotion:12} | r = {r:+.4f} | {strength}"
    
    report += f"""

5. WORST CASE EXAMPLES:
"""
    
    for emotion in ['Fearful', 'Happy', 'Neutral']:
        emotion_col_pred = f'prob_{emotion}'
        emotion_col_voting = EMOTION_COLS[EMOTION_CLASSES.index(emotion)]
        
        residual = predictions_df[emotion_col_pred].values - voting_normalized[emotion_col_voting].values
        worst_idx = np.argmax(np.abs(residual))
        worst_residual = residual[worst_idx]
        worst_filename = predictions_df.iloc[worst_idx]['filename']
        
        report += f"\n  {emotion}:\n"
        report += f"    Worst sample: {worst_filename}\n"
        report += f"    Residual: {worst_residual:+.4f}\n"
        report += f"    {'Over-confident' if worst_residual > 0 else 'Under-confident'}\n"
    
    report += f"""

{'='*80}
INTERPRETATION & IMPLICATIONS
{'='*80}

1. MODEL BIASES:
   - Consistent positive residuals → Model over-confident on that emotion
   - Consistent negative residuals → Model struggles to detect that emotion
   - Large variance → Model predictions for that emotion are unstable

2. CORRELATION INSIGHTS:
   - High r: Model learned discriminative features for this emotion
   - Low r: Model's predictions don't correlate with human rankings
           (absolute error analysis shows it's not about calibration)

3. ALIGNMENT FAILURES:
   - Compare residual distribution to perfect alignment (R=0)
   - If boxplot is far from zero, systematic misalignment exists
   - If whiskers are wide, model lacks robustness

4. LOCALIZATION RISK:
   - Emotions with R << 0 are systematically under-detected
   - Emotions with R >> 0 are systematically over-diagnosed
   - Real-world applications would propagate these biases

5. NEXT STEPS:
   - Identify which emotions need domain adaptation
   - Consider emotion-specific calibration
   - Evaluate need for Taiwanese-specific fine-tuning

{'='*80}
END OF REPORT
{'='*80}
"""
    
    return report

def main():
    print("Starting Residual Analysis...\n")
    
    # Load data
    predictions_df, voting_df = load_data()
    
    # Normalize voting probabilities
    voting_normalized = normalize_voting_probabilities(voting_df)
    
    # Align data (ensure same order)
    # Extract filename without extension for matching
    predictions_df['filename_id'] = predictions_df['filename'].str.replace('.jpg', '').str.replace('.tif', '')
    voting_df['filename_id'] = voting_df['filename'].str.replace('.jpg', '').str.replace('.tif', '')
    
    # Merge on filename
    merged_df = predictions_df.merge(voting_df[['filename_id'] + EMOTION_COLS], 
                                     on='filename_id', how='inner')
    
    # Use normalized voting data for the merged indices
    merged_voting_normalized = voting_normalized.loc[voting_df['filename_id'].isin(merged_df['filename_id'])]
    merged_voting_normalized = merged_voting_normalized.reset_index(drop=True)
    
    # Compute residuals
    residuals = compute_residuals(merged_df, merged_voting_normalized)
    
    # Analyze residuals
    bias_summary = analyze_residuals(residuals)
    
    # Generate visualizations
    plot_residual_boxplots(residuals)
    correlations = plot_correlation_analysis(merged_df, merged_voting_normalized)
    
    # Generate report
    report = generate_summary_report(merged_df, merged_voting_normalized, residuals, bias_summary, correlations)
    
    print(report)
    
    # Save report
    with open('./residual_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✅ Saved: residual_analysis_report.txt")

if __name__ == '__main__':
    main()
