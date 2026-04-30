import torch
from models import VGG16, CUDANotAvailableError
from data import Dataset
from VGG16.utils import ModelManager

CONFIG = {
    'max_epochs': 50,
    'early_stop_patience': 5,
    'lr_schedule': {
        25: 0.001,    # Reduce LR at epoch 25
        35: 0.0001    # Further reduce LR at epoch 35
    },
    'batch_size': 64,
    'num_workers': 10
}


def train(model, trainloader, testloader, device, criterion, optimizer, manager, epochs=100, early_stop_patience=5, lr_schedule=None):
    """Train model with learning rate scheduling, early stopping, and best model saving."""
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Learning rate scheduling
        if lr_schedule and epoch in lr_schedule:
            new_lr = lr_schedule[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f'Learning rate updated to {new_lr}')
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        # Calculate training metrics
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = 100 * correct_train / total_train
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)
        
        # Check if this is the best model
        is_best = val_accuracy > best_val_acc
        if is_best:
            best_val_acc = val_accuracy
            patience_counter = 0
            manager.save_model(model, optimizer, epoch+1, {'val_acc': val_accuracy}, filename='vgg16_best.pth', is_best=True)
        else:
            patience_counter += 1
        
        # Print epoch summary
        print(f'[Epoch {epoch+1}/{epochs}] Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}% | Best: {best_val_acc:.2f}%')
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f'\nEarly stopping triggered! No improvement for {early_stop_patience} epochs.')
            print(f'Best validation accuracy: {best_val_acc:.2f}% at epoch {epoch + 1 - patience_counter}')
            break
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }

if __name__ == "__main__":
    try:
        # Initialize model
        model = VGG16(num_classes=10)
        device = model.device
        criterion = model.criterion
        optimizer = model.optimizer
        
        # Get datasets
        dataset = Dataset(batch_size=CONFIG['batch_size'], num_workers=CONFIG['num_workers'])
        trainloader = dataset.trainloader
        testloader = dataset.testloader
        
        # Initialize model manager
        manager = ModelManager()
        
        # Train the model
        print('Starting training...\n')
        
        results = train(model, trainloader, testloader, device, criterion, optimizer, 
                       manager, epochs=CONFIG["max_epochs"], early_stop_patience=CONFIG["early_stop_patience"],
                         lr_schedule=CONFIG["lr_schedule"])
        
        # Display summary
        print('\n' + '='*60)
        print('TRAINING SUMMARY')
        print('='*60)
        print(f'Final Train Accuracy: {results["train_accuracies"][-1]:.2f}%')
        print(f'Final Val Accuracy: {results["val_accuracies"][-1]:.2f}%')
        print(f'Best Val Accuracy: {results["best_val_acc"]:.2f}%')
        print(f'Total Epochs Trained: {len(results["train_accuracies"])}')
        print('='*60)
        print('\nBest model saved as: saved_models/vgg16_best.pth')
        
        # Save final model
        manager.save_model(model, optimizer, len(results["train_accuracies"]), results, filename='vgg16_final.pth')
        
    except CUDANotAvailableError as cuda_error:
        print('\n' + '='*60)
        print('CUDA ERROR - TRAINING ABORTED')
        print('='*60)
        print(f'{cuda_error}')
        exit(1)
        
    except Exception as e:
        print('\n' + '='*60)
        print('ERROR DURING TRAINING')
        print('='*60)
        print(f'Error: {e}')
        print('='*60)
        import traceback
        traceback.print_exc()
        exit(1)