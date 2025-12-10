import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore")

# Dummy classes for loading old checkpoints with unpicklable objects
class RecorderMeter(object):
    pass

class RecorderMeter1(object):
    pass

try:
    from models.PosterV2_7cls import pyramid_trans_expr2
except ImportError:
    from models.PosterV2_8cls import pyramid_trans_expr2


class TaiwaneseDataset(Dataset):
    """Custom dataset for Taiwanese faces with labels from CSV (voting-based ground truth)
    
    RAF-DB class mapping (alphabetical order):
    0: Angry
    1: Disgusted
    2: Fearful
    3: Happy
    4: Neutral
    5: Sad
    6: Surprised
    """
    
    # Map emotion name to RAF-DB class index
    EMOTION_TO_RAFDB = {
        'Angry': 0,
        'Disgusted': 1,
        'Fearful': 2,
        'Happy': 3,
        'Neutral': 4,
        'Sad': 5,
        'Surprised': 6,
    }
    
    RAFDB_CLASS_NAMES = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
    
    def __init__(self, data_dir, csv_path, transform=None):
        """
        Args:
            data_dir: Path to faces_256x256 folder
            csv_path: Path to taiwanese_ground_truth.csv
            transform: Image transformations
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_list = []
        self.labels = []
        
        # Read CSV file with ground truth from voting
        df = pd.read_csv(csv_path)
        
        print(f"Loaded CSV with columns: {df.columns.tolist()}")
        
        # Load image paths and labels
        for idx, row in df.iterrows():
            img_name = row['filename']
            ground_truth = row['ground_truth']
            
            # Extract filename without extension for matching
            img_name_noext = os.path.splitext(str(img_name))[0]
            
            # Find image with matching ID (ignore extension)
            img_path = None
            for file in os.listdir(self.data_dir):
                if os.path.splitext(file)[0] == img_name_noext:
                    img_path = os.path.join(self.data_dir, file)
                    break
            
            if img_path and ground_truth in self.EMOTION_TO_RAFDB:
                self.image_list.append(img_path)
                # Map to RAF-DB class index (0-6)
                label = self.EMOTION_TO_RAFDB[ground_truth]
                self.labels.append(label)
        
        print(f"Loaded {len(self.image_list)} images from {len(df)} entries")
        print(f"Using RAF-DB class mapping (0-6): {self.RAFDB_CLASS_NAMES}")
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_emotion_name(self, idx):
        """Get emotion name for RAF-DB class index"""
        return self.RAFDB_CLASS_NAMES[idx] if idx < len(self.RAFDB_CLASS_NAMES) else f"Unknown_{idx}"
    
    def get_num_classes(self):
        """Return number of RAF-DB classes (7)"""
        return len(self.RAFDB_CLASS_NAMES)


def main():
    parser = argparse.ArgumentParser(description='Evaluate POSTER V2 on Taiwanese dataset')
    parser.add_argument('--data', type=str, required=True, help='Path to Taiwanese data folder')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
    parser.add_argument('--workers', default=4, type=int, help='Number of workers')
    parser.add_argument('--device', type=str, default='auto', choices=['cuda', 'mps', 'cpu', 'auto'],
                       help='Device to use (auto: cuda if available, else mps if available, else cpu)')
    parser.add_argument('--debug-image', type=str, default=None, help='Image filename to debug (e.g., 0101a02 or 0121b02.jpg)')
    args = parser.parse_args()
    
    # Auto-select device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS (Metal Performance Shaders)")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    else:
        device = torch.device(args.device)
        print(f"Using {args.device.upper()}")
    
    # Define transforms (224x224 resize + normalization)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    csv_path = os.path.join(args.data, 'taiwanese_ground_truth.csv')
    img_dir = os.path.join(args.data, 'faces_256x256')
    
    print(f"Loading dataset from {args.data}")
    print(f"Image directory: {img_dir}")
    print(f"Ground truth CSV: {csv_path}")
    
    dataset = TaiwaneseDataset(img_dir, csv_path, transform=val_transform)
    num_classes = dataset.get_num_classes()
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.workers)
    
    # Load checkpoint to detect model's class count
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        return
    
    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        # Try normal load first
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except (AttributeError, pickle.UnpicklingError) as e:
        # If checkpoint has unpicklable objects, try loading with pickle protocol
        print(f"Warning: Checkpoint has extra objects, attempting alternative load...")
        import pickle
        with open(args.checkpoint, 'rb') as f:
            checkpoint = pickle.load(f)
    
    # Handle DataParallel checkpoint
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        print("ERROR: Could not extract state_dict from checkpoint")
        return
    
    # Detect model's number of classes from checkpoint
    model_num_classes = None
    for k, v in state_dict.items():
        if 'classifier' in k or 'head' in k or 'linear' in k:
            if 'weight' in k:
                model_num_classes = v.shape[0]
                print(f"Detected {model_num_classes} classes from {k}")
                break
    
    if model_num_classes is None:
        print(f"WARNING: Could not detect model's class count, assuming 7")
        model_num_classes = 7
    
    print(f"Model classes: {model_num_classes}, Dataset classes: {num_classes}")
    
    # Load model with detected class count
    model = pyramid_trans_expr2(img_size=224, num_classes=model_num_classes)
    
    # Remove 'module.' prefix if exists
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    print("Checkpoint loaded successfully!")
    
    model = model.to(device)
    model.eval()
    
    # Debug mode: single image
    if args.debug_image:
        print(f"\n{'='*60}")
        print(f"DEBUG MODE: {args.debug_image}")
        print(f"{'='*60}")
        
        # RAF-DB to alphabetical mapping (same as in evaluation)
        RAFDB_TO_ALPHA_MAPPING = {
            0: 6,  # Surprise -> Surprised
            1: 2,  # Fear -> Fearful
            2: 1,  # Disgust -> Disgusted
            3: 3,  # Happiness -> Happy
            4: 5,  # Sadness -> Sad
            5: 0,  # Anger -> Angry
            6: 4   # Neutral -> Neutral
        }
        
        # Find the image in dataset by ID (ignore extension)
        debug_id = os.path.splitext(args.debug_image)[0]
        found = False
        for idx in range(len(dataset.image_list)):
            img_id = os.path.splitext(os.path.basename(dataset.image_list[idx]))[0]
            if img_id == debug_id:
                found = True
                img_path = dataset.image_list[idx]
                ground_truth = dataset.labels[idx]
                
                # Load and process image
                image = Image.open(img_path).convert('RGB')
                image_tensor = val_transform(image).unsqueeze(0).to(device)
                
                # Get prediction
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]
                    pred_idx_raw = torch.argmax(probabilities).item()
                    
                    # Remap from RAF-DB to alphabetical order
                    pred_idx = RAFDB_TO_ALPHA_MAPPING[pred_idx_raw]
                    
                    # Remap probabilities
                    probs_remapped = torch.zeros_like(probabilities)
                    for old_idx, new_idx in RAFDB_TO_ALPHA_MAPPING.items():
                        probs_remapped[new_idx] = probabilities[old_idx]
                
                # Display results
                print(f"Image path: {img_path}")
                print(f"Ground truth: {dataset.get_emotion_name(ground_truth)} (index {ground_truth})")
                print(f"Prediction: {dataset.get_emotion_name(pred_idx)} (index {pred_idx})")
                print(f"Confidence: {probs_remapped[pred_idx]:.4f}")
                print(f"\nAll class probabilities (remapped to alphabetical order):")
                for class_idx in range(len(dataset.RAFDB_CLASS_NAMES)):
                    class_name = dataset.get_emotion_name(class_idx)
                    print(f"  {class_name:15s}: {probs_remapped[class_idx]:.4f}")
                print(f"{'='*60}\n")
                break
        
        if not found:
            print(f"ERROR: Image '{args.debug_image}' not found in dataset")
        return
    
    # Evaluation
    all_preds = []
    all_probs = []  # Store all probability vectors
    all_labels = []
    all_filenames = []  # Store filenames for CSV
    
    # RAF-DB model output index to alphabetical order mapping
    # Model (POSTER V2 RAF-DB): 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
    # Target (Alphabetical): 0:Angry, 1:Disgusted, 2:Fearful, 3:Happy, 4:Neutral, 5:Sad, 6:Surprised
    RAFDB_TO_ALPHA_MAPPING = {
        0: 6,  # Surprise -> Surprised
        1: 2,  # Fear -> Fearful
        2: 1,  # Disgust -> Disgusted
        3: 3,  # Happiness -> Happy
        4: 5,  # Sadness -> Sad
        5: 0,  # Anger -> Angry
        6: 4   # Neutral -> Neutral
    }
    
    print("\nRunning inference...")
    print("Note: Remapping model outputs from RAF-DB order to alphabetical order...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probabilities, 1)
            
            # Remap predictions from RAF-DB order to alphabetical order
            preds_remapped = torch.tensor([RAFDB_TO_ALPHA_MAPPING[p.item()] for p in preds])
            
            # Remap probabilities matrix
            probs_remapped = probabilities.cpu().numpy().copy()
            for old_idx, new_idx in RAFDB_TO_ALPHA_MAPPING.items():
                probs_remapped[:, new_idx] = probabilities.cpu().numpy()[:, old_idx]
            
            all_preds.extend(preds_remapped.numpy())
            all_probs.extend(probs_remapped)
            all_labels.extend(labels.cpu().numpy())
            
            # Get filenames from dataset (match batch order)
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(dataset.image_list))
            for i in range(start_idx, end_idx):
                filename = os.path.basename(dataset.image_list[i])
                all_filenames.append(filename)
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total samples: {len(all_labels)}")
    print(f"Number of classes: {num_classes}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1-Score (weighted): {f1:.4f}")
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    emotion_names = [dataset.get_emotion_name(i) for i in range(num_classes)]
    print(classification_report(all_labels, all_preds, target_names=emotion_names))
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Print per-class accuracy
    print("\n" + "="*60)
    print("PER-CLASS ACCURACY")
    print("="*60)
    for i in range(num_classes):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == i).sum() / mask.sum()
            emotion_name = dataset.get_emotion_name(i)
            count = mask.sum()
            print(f"{emotion_name}: {class_acc:.4f} ({class_acc*100:.2f}%) [{count} samples]")
    
    # Save results to CSV
    print("\n" + "="*60)
    print("SAVING RESULTS TO CSV")
    print("="*60)
    
    # Create results dataframe
    results_data = {
        'filename': all_filenames,
        'ground_truth': [dataset.get_emotion_name(int(label)) for label in all_labels],
        'ground_truth_idx': all_labels,
        'prediction': [dataset.get_emotion_name(int(pred)) for pred in all_preds],
        'prediction_idx': all_preds,
        'correct': (all_labels == all_preds).astype(int),
    }
    
    # Add probability columns for each class
    for class_idx in range(num_classes):
        class_name = dataset.get_emotion_name(class_idx)
        results_data[f'prob_{class_name}'] = all_probs[:, class_idx]
    
    results_df = pd.DataFrame(results_data)
    
    # Save to CSV with timestamp
    output_csv = os.path.join(args.data, f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    results_df.to_csv(output_csv, index=False)
    print(f"✅ Saved predictions to: {output_csv}")
    print(f"   Total predictions: {len(results_df)}")
    print(f"   Correct: {results_df['correct'].sum()} ({results_df['correct'].mean()*100:.2f}%)")


if __name__ == '__main__':
    main()
