import torch
import os
from pathlib import Path

class ModelManager:
    """Handle model saving, loading, and testing."""
    
    def __init__(self, model_dir='saved_models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
    
    def save_model(self, model, optimizer, epoch, results, filename='vgg16_model.pth', is_best=False):
        """Save model checkpoint with training results."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results': results
        }
        filepath = self.model_dir / filename
        torch.save(checkpoint, filepath)
        status = '(BEST)' if is_best else ''
        print(f'Model saved to {filepath} {status}')
        return filepath
    
    def load_model(self, model, optimizer, filename='vgg16_model.pth'):
        """Load model checkpoint."""
        filepath = self.model_dir / filename
        
        if not filepath.exists():
            print(f'Model file not found: {filepath}')
            return None
        
        checkpoint = torch.load(filepath, map_location=model.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        results = checkpoint.get('results', {})
        
        print(f'Model loaded from {filepath} (Epoch {epoch})')
        return epoch, results
    
    def test_model(self, model, testloader, device):
        """Test model on test set and return accuracy."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'✓ Test Accuracy: {accuracy:.2f}% ({correct}/{total})')
        return accuracy
    
    def get_model_info(self, model):
        """Print model architecture information."""
        print('\n' + '='*60)
        print('MODEL ARCHITECTURE')
        print('='*60)
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\nTotal Parameters: {total_params:,}')
        print(f'Trainable Parameters: {trainable_params:,}')
        print('='*60 + '\n')
