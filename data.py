import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class Dataset:
    def __init__(self, batch_size=32, num_workers=2):
        # Normalization constants for CIFAR-10
        self.normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        
        # Training transforms with MINIMAL augmentation (testing if augmentation is the issue)
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            self.normalize
        ])
        
        # Testing transforms (no augmentation)
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])
        
        # Load training set with augmentation
        self.trainset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=True, 
            download=True, 
            transform=self.train_transform
        )
        print(f" Training set loaded: {len(self.trainset)} images")
        self.trainloader = DataLoader(
            self.trainset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        
        # Load test set without augmentation
        self.testset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=False, 
            download=True, 
            transform=self.test_transform
        )
        print(f"✓ Test set loaded: {len(self.testset)} images")
        self.testloader = DataLoader(
            self.testset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
    