import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import snntorch.functional as SF

from model import AudioSNN
from data_loader import get_speech_commands_loaders


def train_epoch(model, train_loader, optimizer, loss_fn, device):

    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, targets in tqdm(train_loader, desc="Training", leave=False):
        data = data.to(device)


        spk_rec, _ = model(data) 
        
        loss = loss_fn(spk_rec, targets) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        correct += SF.accuracy_rate(spk_rec, targets) * targets.size(0)
        total += targets.size(0)
        
        total_loss += loss.item()
        _, predicted = spk_rec.sum(dim=0).max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, test_loader, loss_fn, device):

    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc="Testing", leave=False):
            data = data.to(device)
            targets = targets.to(device)
            
            spk_rec, mem_rec = model(data)
            
            loss = loss_fn(mem_rec, targets)
            
            total_loss += loss.item()
            _, predicted = spk_rec.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def plot_metrics(train_losses, test_losses, train_accs, test_accs, save_path='metrics.png'):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Testing Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(test_accs, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Testing Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Metrics plot saved to {save_path}")


def train(num_epochs=15, batch_size=32, lr=0.001, beta=0.95, 
          num_steps=25, n_mels=64, target_length=32, 
          save_dir='checkpoints', seed=42):

    torch.manual_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading Speech Commands dataset...")
    train_loader, test_loader, classes = get_speech_commands_loaders(
        batch_size=batch_size,
        n_mels=n_mels,
        target_length=target_length
    )
    
    print(f"Classes: {classes}")
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of testing samples: {len(test_loader.dataset)}")
    
    print("Creating model...")
    model = AudioSNN(
        n_mels=n_mels, 
        seq_length=target_length,
        num_classes=len(classes),
        beta=beta, 
        num_steps=num_steps
    ).to(device)
    
    loss_fn = SF.ce_rate_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    print(f"\nStarting training for {num_epochs} epochs...")
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'classes': classes,
            }, checkpoint_path)
            print(f"Best model saved with accuracy: {best_acc:.2f}%")
    
    final_path = os.path.join(save_dir, 'final_model.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'classes': classes,
    }, final_path)
    
    plot_metrics(train_losses, test_losses, train_accs, test_accs, 
                 os.path.join(save_dir, 'metrics.png'))
    
    print(f"\nTraining complete! Best test accuracy: {best_acc:.2f}%")
    

if __name__ == '__main__':
    train()
