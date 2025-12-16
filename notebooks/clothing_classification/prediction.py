from pathlib import Path
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn

# Define classes (same as training)
CLASSES = ['Light clothing', 'Medium clothing', 'Heavy clothing']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the same transform used during validation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_clothing(image_path, model_path='best_resnet50_clothing.pth'):
    """
    Predict clothing category from image
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Recreate model architecture
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(CLASSES))
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(DEVICE)
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_tensor = val_transform(img).unsqueeze(0).to(DEVICE)
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)
    
    return {
        'class': CLASSES[predicted.item()],
        'confidence': confidence.item() * 100,
        'all_probabilities': {
            CLASSES[i]: probabilities[0][i].item() * 100
            for i in range(len(CLASSES))
        }
    }

# Example usage
result = predict_clothing('path/to/your/image.jpg')

print(f"Predicted Class: {result['class']}")
print(f"Confidence: {result['confidence']:.2f}%")
print("\nAll Probabilities:")
for cls, prob in result['all_probabilities'].items():
    print(f"  {cls}: {prob:.2f}%")