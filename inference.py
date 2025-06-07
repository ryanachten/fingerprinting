import torch
import ml_models
from torchvision import transforms
from PIL import Image
import numpy as np
from fingerprinting import random_crop_tensor_test
import argparse
import os
from collections import Counter

def analyze_multiple_regions(image, model, device, num_regions=3):
    """Analyze multiple regions of the image and vote on the final prediction"""
    predictions = []
    confidences = []
    all_probs = []
    
    for i in range(num_regions):
        # Process different regions of the image
        transform = transforms.Compose([
            random_crop_tensor_test(scale=2, test_samples=12, scale_2=1),
            transforms.Normalize([0.0895, 0.0895, 0.0895], [0.1101, 0.1101, 0.1101])
        ])
        
        try:
            inputs = transform(image).to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
                outputs = outputs.view(12, -1, 21)
                
                # Sum logits across test samples
                region_output = outputs.sum(0)
                probabilities = torch.nn.functional.softmax(region_output, dim=1)
                score, pred = torch.max(probabilities, 1)
                
                predictions.append(pred.item())
                confidences.append(score.item())
                all_probs.append(probabilities)
                
        except Exception as e:
            print(f"Warning: Error processing region {i}: {e}")
            continue
    
    if not predictions:
        return None, 0, None
    
    # Get the most common prediction
    pred_counter = Counter(predictions)
    final_pred = pred_counter.most_common(1)[0][0]
    
    # Calculate average probability distribution
    avg_probs = torch.mean(torch.cat([p for p in all_probs]), dim=0)
    
    # Calculate confidence as percentage of regions with this prediction
    final_confidence = pred_counter[final_pred] / len(predictions)
    
    return final_pred, final_confidence, avg_probs

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Predict printer model from an image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--model_path', type=str, 
                        default='./data/Models/fingerprinting_21_printer_best_model_efficientnetv2_m.pth',
                        help='Path to the model weights file')
    parser.add_argument('--regions', type=int, default=3,
                        help='Number of image regions to analyze for voting')
    args = parser.parse_args()

    # Check if image file exists
    if not os.path.isfile(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found")
        return

    # 1. Load the model with the correct configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_result = ml_models.initialize_model(
        model_name='efficientnetv2_m',
        num_classes=21,
        feature_extract=False,
        freeze_layers=False,
        classifier_layer_config=0,
        input_size=448
    )

    model = model_result[0]

    # Reset classifier to match saved weights
    if hasattr(model, 'classifier') and isinstance(model.classifier, torch.nn.Sequential):
        model.classifier[1] = torch.nn.Sequential(
            torch.nn.Linear(1280, 21)
        )

    # 2. Load pre-trained weights
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded successfully from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device)
    model.eval()

    # 3. Run inference with multi-region analysis
    try:
        image = Image.open(args.image_path).convert('RGB')
        print(f"Image loaded: {args.image_path} (size: {image.size})")
        
        # Analyze multiple regions for more robust prediction
        pred, confidence, avg_probs = analyze_multiple_regions(
            image, model, device, num_regions=args.regions
        )
        
        if pred is None:
            print("Failed to analyze image. Try a different image or check for errors.")
            return
            
    except Exception as e:
        print(f"Error during inference: {e}")
        return

    # 4. Map prediction to printer model
    printer_models = [
        "Stratasys450mc-2", "5200-1", "Stratasys450mc-1", "Stratasys900mc-1",
        "M2-1", "Form3B-4", "Form3B-2", "Form3B-1", "Form3B-5",
        "Form3B-6", "Form3B-3", "4200-1", "5200-2", "L1-2",
        "L1-1", "M2-2", "M2-3", "M2-4", "M2-5", "Stratasys900mc-2", "M2-6"
    ]

    # 5. Output results with comprehensive details
    print(f"\nResults for {args.image_path}:")
    print(f"Predicted printer: {printer_models[pred]}")
    print(f"Confidence score: {confidence:.4f} ({confidence*100:.2f}%)")
    
    # Display top-k predictions where k is min(5, number of classes)
    # Safety check for avg_probs dimensions
    if avg_probs is not None:
        # Check tensor dimension and shape properly
        num_classes = avg_probs.size(0)  # For 1D tensor, use dimension 0
        k = min(5, num_classes)
        
        # Get topk from the tensor with proper dimensionality
        values, indices = torch.topk(avg_probs, k)
        
        print(f"\nTop {k} predictions:")
        for i in range(k):
            print(f"  {i+1}. {printer_models[indices[i]]} - {values[i]*100:.2f}%")
    else:
        print("\nCould not generate detailed probability distribution.")
    
    # If confidence is low, provide a warning and suggestions
    if confidence < 0.7:
        print("\nWarning: Low confidence prediction. Consider:")
        print("- Using a clearer image with more printer-specific patterns")
        print("- Ensuring the image has sufficient resolution")
        print("- Using more regions for analysis (--regions 5)")

if __name__ == "__main__":
    main()