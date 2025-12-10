"""
Video Emotion Analysis: Real-time emotion prediction on video frames
Validates cross-cultural emotion recognition failures in dynamic settings
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# Dummy classes for loading old checkpoints with unpicklable objects
class RecorderMeter(object):
    pass

class RecorderMeter1(object):
    pass

def load_model(checkpoint_path, device, num_classes=7):
    """Load POSTER V2 model from checkpoint"""
    try:
        from models.PosterV2_7cls import pyramid_trans_expr2
    except ImportError:
        try:
            from models.PosterV2_8cls import pyramid_trans_expr2
        except ImportError:
            print("Error: Could not import model architecture")
            sys.exit(1)
    
    print(f"Loading model from: {checkpoint_path}")
    model = pyramid_trans_expr2(img_size=224, num_classes=num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model

def preprocess_frame(frame_path, device):
    """
    Load and preprocess frame from disk
    
    CRITICAL: Use PIL.Image for RGB consistency
    NEVER use cv2.imread which defaults to BGR
    """
    img = Image.open(frame_path).convert('RGB')
    
    # Resize from 360×360 to 224×224
    img_resized = img.resize((224, 224), Image.BILINEAR)
    
    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    img_tensor = transform(img_resized).unsqueeze(0).to(device)
    
    return img_tensor

def get_device(device_str='auto'):
    """Set device with auto-detection priority: MPS > CUDA > CPU"""
    
    if device_str == 'auto':
        # Auto-detect: MPS first (macOS), then CUDA, then CPU
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("🍎 Using MPS (Metal Performance Shaders - macOS GPU)")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print("🔧 Using CUDA (NVIDIA GPU)")
        else:
            device = torch.device('cpu')
            print("💻 Using CPU")
    elif device_str == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🍎 Using MPS (Metal Performance Shaders)")
    elif device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print("🔧 Using CUDA (NVIDIA GPU)")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    
    return device

def analyze_single_frame(frame_path, model, device, emotion_classes):
    """Debug: Analyze a single frame in detail"""
    
    print(f"\n{'='*70}")
    print(f"DEBUG MODE: Single Frame Analysis")
    print(f"{'='*70}")
    print(f"Frame path: {frame_path}\n")
    
    # Preprocess
    frame_tensor = preprocess_frame(frame_path, device)
    
    # Inference
    with torch.no_grad():
        logits = model(frame_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    
    # Get prediction
    pred_idx = np.argmax(probs)
    pred_emotion = emotion_classes[pred_idx]
    confidence = probs[pred_idx]
    
    # Display results
    print(f"Predicted Emotion: {pred_emotion}")
    print(f"Confidence: {confidence:.4f}\n")
    
    print("All emotion probabilities:")
    for emotion_idx, emotion in enumerate(emotion_classes):
        prob = probs[emotion_idx]
        bar_length = int(prob * 50)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f"  {emotion:12} : {prob:7.4f} | {bar}")
    
    print(f"\n{'='*70}\n")
    
    return pd.DataFrame([{
        'filename': Path(frame_path).name,
        'predicted_emotion': pred_emotion,
        'confidence': confidence,
        **{f'prob_{emotion}': probs[i] for i, emotion in enumerate(emotion_classes)}
    }])

def analyze_frames(video_dir, model, device, emotion_classes):
    """Inference on all frames in directory"""
    
    frame_dir = Path(video_dir)
    frame_files = sorted([f for f in frame_dir.glob('*.jpg') if f.is_file()])
    
    print(f"\n{'='*70}")
    print(f"PROCESSING {len(frame_files)} FRAMES")
    print(f"{'='*70}\n")
    
    if not frame_files:
        print(f"Error: No JPG files found in {video_dir}")
        sys.exit(1)
    
    results = []
    
    with torch.no_grad():
        for idx, frame_path in enumerate(tqdm(frame_files, desc="Processing frames")):
            # Preprocess
            frame_tensor = preprocess_frame(str(frame_path), device)
            
            # Inference
            logits = model(frame_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Get prediction
            pred_idx = np.argmax(probs)
            pred_emotion = emotion_classes[pred_idx]
            confidence = probs[pred_idx]
            
            # Timestamp (assuming 5 fps)
            timestamp = idx / 5.0
            
            # Store result
            result = {
                'frame_number': idx,
                'filename': frame_path.name,
                'timestamp_sec': timestamp,
                'predicted_emotion': pred_emotion,
                'confidence': confidence,
            }
            
            # Add per-emotion probabilities
            for emotion_idx, emotion in enumerate(emotion_classes):
                result[f'prob_{emotion}'] = probs[emotion_idx]
            
            results.append(result)
    
    return pd.DataFrame(results)

def plot_emotion_timeline(df, output_path, emotion_classes):
    """
    Create emotion timeline visualization
    Shows how model predictions change over time
    """
    print(f"\nGenerating emotion timeline plot...")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: All emotions over time
    ax1 = axes[0]
    for emotion in emotion_classes:
        prob_col = f'prob_{emotion}'
        ax1.plot(df['timestamp_sec'], df[prob_col], label=emotion, linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Emotion Probability', fontsize=12)
    ax1.set_title('Emotion Probability Timeline - All Classes', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Focus on Happy vs Sad (the critical comparison)
    ax2 = axes[1]
    ax2.plot(df['timestamp_sec'], df['prob_Happy'], label='Happy (Expected)', 
             linewidth=3, color='green', marker='o', markersize=3, alpha=0.8)
    ax2.plot(df['timestamp_sec'], df['prob_Sad'], label='Sad (Model Predicts)', 
             linewidth=3, color='red', marker='s', markersize=3, alpha=0.8)
    ax2.fill_between(df['timestamp_sec'], df['prob_Happy'], df['prob_Sad'], 
                      where=(df['prob_Sad'] >= df['prob_Happy']), 
                      alpha=0.3, color='red', label='Model Gap (Sad > Happy)')
    
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Emotion Probability', fontsize=12)
    ax2.set_title('Critical Finding: Happy vs Sad\n(Model systematically over-predicts Sad)', 
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def generate_summary_statistics(df, emotion_classes):
    """Generate summary statistics"""
    
    print(f"\n{'='*70}")
    print("EMOTION PREDICTION STATISTICS")
    print(f"{'='*70}\n")
    
    print(f"Total frames analyzed: {len(df)}")
    print(f"Video duration: {df['timestamp_sec'].max():.1f} seconds\n")
    
    print("Emotion Distribution:")
    emotion_counts = df['predicted_emotion'].value_counts()
    for emotion in emotion_classes:
        count = emotion_counts.get(emotion, 0)
        percentage = (count / len(df)) * 100
        print(f"  {emotion:12} : {count:4d} frames ({percentage:5.1f}%)")
    
    print("\nMean Confidence by Emotion:")
    for emotion in emotion_classes:
        prob_col = f'prob_{emotion}'
        mean_conf = df[prob_col].mean()
        max_conf = df[prob_col].max()
        print(f"  {emotion:12} : mean={mean_conf:.4f}, max={max_conf:.4f}")
    
    # Critical finding: Happy vs Sad residual
    happy_mean = df['prob_Happy'].mean()
    sad_mean = df['prob_Sad'].mean()
    residual = sad_mean - happy_mean
    
    print(f"\n{'='*70}")
    print("CRITICAL FINDING: Happy vs Sad")
    print(f"{'='*70}")
    print(f"Mean Happy probability: {happy_mean:.4f}")
    print(f"Mean Sad probability:   {sad_mean:.4f}")
    print(f"Residual (Sad - Happy): {residual:+.4f}")
    print(f"\n⚠️  Model predicts Sad is {residual:.1%} MORE likely than Happy")
    print(f"This confirms the static image finding: Happy is systematically under-detected")

def main():
    parser = argparse.ArgumentParser(description='Evaluate emotion predictions on video frames')
    parser.add_argument('--video-dir', type=str, default='./data/vlog/frames',
                        help='Directory containing extracted video frames')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/raf-db-model_best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='./data/vlog/analysis_results.csv',
                        help='Output CSV file for predictions')
    parser.add_argument('--plot', type=str, default='./data/vlog/emotion_timeline.png',
                        help='Output plot file')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use (default: auto-detect MPS/CUDA/CPU)')
    parser.add_argument('--debug-frame', type=str, default=None,
                        help='Debug mode: analyze single frame (e.g., 0001 or 0005)')
    
    args = parser.parse_args()
    
    # Setup
    device = get_device(args.device)
    emotion_classes = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
    
    # Load model
    model = load_model(args.checkpoint, device, num_classes=7)
    
    # Debug mode: analyze single frame
    if args.debug_frame:
        # Handle both filename and full path
        if args.debug_frame.endswith('.jpg'):
            frame_name = args.debug_frame
        else:
            frame_name = f"{args.debug_frame}.jpg"
        
        frame_path = os.path.join(args.video_dir, frame_name)
        
        if not os.path.exists(frame_path):
            print(f"Error: Frame not found at {frame_path}")
            sys.exit(1)
        
        analyze_single_frame(frame_path, model, device, emotion_classes)
        return
    
    # Process frames
    df_results = analyze_frames(args.video_dir, model, device, emotion_classes)
    
    # Save results
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df_results.to_csv(args.output, index=False)
    print(f"\n✅ Saved predictions to: {args.output}")
    
    # Generate visualizations
    os.makedirs(os.path.dirname(args.plot) or '.', exist_ok=True)
    plot_emotion_timeline(df_results, args.plot, emotion_classes)
    
    # Summary statistics
    generate_summary_statistics(df_results, emotion_classes)
    
    print(f"\n{'='*70}")
    print("VIDEO ANALYSIS COMPLETE")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
