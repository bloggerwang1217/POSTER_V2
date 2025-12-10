"""
Visualize video emotion analysis results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def visualize_emotion_timeline(csv_path, output_dir='./analysis'):
    """Create comprehensive emotion timeline visualization"""
    
    df = pd.read_csv(csv_path)
    emotion_classes = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
    
    print(f"Loading {len(df)} frames from {csv_path}")
    print(f"Duration: {df['timestamp_sec'].max():.1f} seconds")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # ========== Plot 1: All emotions timeline ==========
    ax1 = plt.subplot(3, 1, 1)
    colors = ['#FF6B6B', '#FFA500', '#4ECDC4', '#FFD93D', '#95E1D3', '#C44569', '#9B59B6']
    
    for emotion_idx, emotion in enumerate(emotion_classes):
        prob_col = f'prob_{emotion}'
        ax1.plot(df['timestamp_sec'], df[prob_col], 
                label=emotion, linewidth=2.5, alpha=0.8, color=colors[emotion_idx], marker='o', markersize=2)
    
    ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax1.set_title('All Emotions Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', ncol=4, fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 0.25])
    
    # ========== Plot 2: Model Confidence (How sure is it?) ==========
    ax2 = plt.subplot(3, 1, 2)
    
    ax2.fill_between(df['timestamp_sec'], 0, df['confidence'], alpha=0.4, color='purple', label='Model Confidence')
    ax2.plot(df['timestamp_sec'], df['confidence'], linewidth=2.5, color='purple', marker='o', markersize=4, alpha=0.8)
    
    # Random chance line
    random_chance = 1.0 / 7  # 7 classes
    ax2.axhline(y=random_chance, color='red', linestyle='--', linewidth=2, label=f'Random Chance ({random_chance:.4f})')
    
    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Confidence', fontsize=12, fontweight='bold')
    ax2.set_title('Model Confidence vs Random Chance (14.3%)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 0.25])
    
    # Add text annotation
    conf_mean = df['confidence'].mean()
    conf_std = df['confidence'].std()
    
    textstr = f'Mean Confidence: {conf_mean:.4f}\nStd Dev: {conf_std:.4f}\nRandom: {random_chance:.4f}'
    ax2.text(0.02, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ========== Plot 3: Predicted emotion over time (bar chart) ==========
    ax3 = plt.subplot(3, 1, 3)
    
    emotion_to_idx = {emotion: i for i, emotion in enumerate(emotion_classes)}
    predicted_indices = [emotion_to_idx[e] for e in df['predicted_emotion']]
    
    bars = ax3.bar(df['timestamp_sec'], predicted_indices, width=0.15, color='steelblue', alpha=0.7)
    
    ax3.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Predicted Emotion', fontsize=12, fontweight='bold')
    ax3.set_title('Predicted Emotion (Discrete Classification)', fontsize=14, fontweight='bold')
    ax3.set_yticks(range(len(emotion_classes)))
    ax3.set_yticklabels(emotion_classes)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    # Save main plot
    output_path = Path(output_dir) / 'emotion_timeline_full.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()
    
    # ========== Additional: Confidence distribution ==========
    fig2, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Confidence over time
    ax_conf = axes[0]
    ax_conf.plot(df['timestamp_sec'], df['confidence'], linewidth=2.5, color='purple', marker='o', markersize=3, alpha=0.7)
    ax_conf.fill_between(df['timestamp_sec'], 0, df['confidence'], alpha=0.3, color='purple')
    ax_conf.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax_conf.set_ylabel('Model Confidence', fontsize=12, fontweight='bold')
    ax_conf.set_title('Model Confidence Over Time', fontsize=13, fontweight='bold')
    ax_conf.grid(True, alpha=0.3, linestyle='--')
    ax_conf.set_ylim([0, 0.3])
    
    # Emotion distribution histogram
    ax_hist = axes[1]
    emotion_counts = df['predicted_emotion'].value_counts()
    emotion_counts = emotion_counts.reindex(emotion_classes, fill_value=0)
    
    bars = ax_hist.bar(emotion_classes, emotion_counts.values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax_hist.set_ylabel('Frame Count', fontsize=12, fontweight='bold')
    ax_hist.set_title('Emotion Prediction Distribution', fontsize=13, fontweight='bold')
    ax_hist.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax_hist.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save confidence plot
    output_path2 = Path(output_dir) / 'emotion_confidence_distribution.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path2}")
    plt.close()
    
    # ========== Summary Statistics ==========
    print(f"\n{'='*70}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"Total frames: {len(df)}")
    print(f"Duration: {df['timestamp_sec'].max():.1f} seconds")
    print(f"Frame rate: 5 fps (assumed)\n")
    
    print("Emotion Prediction Distribution:")
    for emotion in emotion_classes:
        count = (df['predicted_emotion'] == emotion).sum()
        pct = (count / len(df)) * 100
        print(f"  {emotion:12} : {count:3d} frames ({pct:5.1f}%)")
    
    print(f"\nConfidence Statistics:")
    print(f"  Mean confidence : {df['confidence'].mean():.4f}")
    print(f"  Min confidence  : {df['confidence'].min():.4f}")
    print(f"  Max confidence  : {df['confidence'].max():.4f}")
    
    print(f"\nEmotion Probability Statistics (per emotion):")
    for emotion in emotion_classes:
        prob_col = f'prob_{emotion}'
        mean_prob = df[prob_col].mean()
        max_prob = df[prob_col].max()
        print(f"  {emotion:12} : mean={mean_prob:.4f}, max={max_prob:.4f}")
    
    # Critical finding
    conf_mean = df['confidence'].mean()
    random_chance = 1.0 / 7
    
    print(f"\n{'='*70}")
    print("CRITICAL FINDING: Uniform Confusion (Complete Model Failure)")
    print(f"{'='*70}")
    print(f"Mean model confidence: {conf_mean:.4f}")
    print(f"Random chance (1/7):   {random_chance:.4f}")
    print(f"Difference:            {conf_mean - random_chance:+.4f}")
    
    if conf_mean < random_chance * 1.2:
        print(f"\n⚠️  Model confidence is BARELY ABOVE random guessing!")
        print(f"All 7 emotions have nearly equal probability (~14%)")
        print(f"This indicates: The model has learned NO meaningful features")
        print(f"for this video (extreme cultural/domain mismatch)")
    elif conf_mean < random_chance * 1.5:
        print(f"\n⚠️  Model confidence is still very close to random")
        print(f"Slight preference for certain emotions, but essentially useless")
    
    # Check entropy
    entropy_per_frame = []
    for _, row in df.iterrows():
        probs = [row[f'prob_{e}'] for e in emotion_classes]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        entropy_per_frame.append(entropy)
    
    mean_entropy = np.mean(entropy_per_frame)
    max_entropy = np.log(7)  # 7 classes
    
    print(f"\nShannon Entropy Analysis:")
    print(f"  Mean entropy:        {mean_entropy:.4f}")
    print(f"  Max entropy (7 cls): {max_entropy:.4f} (uniform distribution)")
    print(f"  Ratio to max:        {mean_entropy/max_entropy*100:.1f}%")
    
    if mean_entropy > max_entropy * 0.9:
        print(f"\n💥 NEAR-MAXIMUM ENTROPY: Predictions are almost perfectly uniform!")
        print(f"This is THE smoking gun proving model failure")
    
    print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(description='Visualize video emotion analysis results')
    parser.add_argument('--csv', type=str, default='./data/vlog/analysis_results.csv',
                        help='Path to analysis results CSV')
    parser.add_argument('--output', type=str, default='./data/vlog',
                        help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    visualize_emotion_timeline(args.csv, args.output)

if __name__ == '__main__':
    main()
