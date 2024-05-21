import numpy as np
import torch
from torcheval.metrics.functional import multiclass_accuracy, multiclass_precision, multiclass_recall, multiclass_auroc
import json
from tqdm import tqdm

def Train(model, max_epochs, train_data, val_data, loss_func, optimiser, scheduler=None, early_stopping=None, metrics_path='./metrics.json', save_path='./best-model.pth'):
    # Check if can run on cuda
    if torch.cuda.is_available():
        device = 'cuda'
        model.cuda()
    else:
        device = 'cpu'
    # Tell user what device running on
    print(f'Running on {device}')
    # Default best accuracy and patience
    best_val_acc = 0
    patience = 0
    # Dictionary to keep track of metrics
    metrics = {
    'train_loss_per_epoch':[],
    'train_acc_per_epoch':[],
    'train_prec_per_epoch':[],
    'train_recall_per_epoch':[],
    'train_auroc_per_epoch':[],
    'val_loss_per_epoch':[],
    'val_acc_per_epoch':[],
    'val_prec_per_epoch':[],
    'val_recall_per_epoch':[],
    'val_auroc_per_epoch':[]
    }

    for epoch in range(max_epochs):
        train_loss = []
        train_acc = []
        train_prec = []
        train_recall = []
        train_auroc = []
        print(f'=== Epoch: {epoch + 1}/{max_epochs} ===')
        # Put model into train mode
        model.train()
        # Training
        for image, label in tqdm(train_data):
            # Move to device
            image, label = image.to(device), label.to(device)
            # Reset gradient
            optimiser.zero_grad()
            # Make predictions
            pred = model(image)
            # Calculate loss
            loss = loss_func(pred, label)
            # Backpropagation
            loss.backward()
            optimiser.step()
            # Calculate metrics
            acc = multiclass_accuracy(pred, label)
            prec = multiclass_precision(pred, label)
            recall = multiclass_recall(pred, label)
            auroc = multiclass_auroc(pred, label, num_classes=7)
            # Update metric arrays
            train_loss.append(loss.item())
            train_acc.append(acc.item())
            train_prec.append(prec.item())
            train_recall.append(recall.item())
            train_auroc.append(auroc.item())
        # Update scheduler if using
        if scheduler is not None:
            prev_lr = optimiser.param_groups[0]['lr']
            scheduler.step()
            new_lr = optimiser.param_groups[0]['lr']
            print(f'= Learning Rate {prev_lr} --> {new_lr} =')

        # Get average metrics per epoch
        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_train_acc = sum(train_acc) / len(train_acc)
        avg_train_prec = sum(train_prec) / len(train_prec)
        avg_train_recall = sum(train_recall) / len(train_recall)
        avg_train_auroc = sum(train_auroc) / len(train_auroc)
        metrics['train_loss_per_epoch'].append(avg_train_loss)
        metrics['train_acc_per_epoch'].append(avg_train_acc)
        metrics['train_prec_per_epoch'].append(avg_train_prec)
        metrics['train_recall_per_epoch'].append(avg_train_recall)
        metrics['train_auroc_per_epoch'].append(avg_train_auroc)

        # Validation
        # Put model into eval mode
        model.eval()
        val_loss = []
        val_acc = []
        val_prec = []
        val_recall = []
        val_auroc = []

        for image, label in tqdm(val_data):
            # Move to device
            image, label = image.to(device), label.to(device)
            # Don't calculate gradient
            with torch.no_grad():
                # Make predictions
                pred = model(image)
                # Calculate loss
                loss = loss_func(pred, label)
                # Calculate metrics
                acc = multiclass_accuracy(pred, label)
                prec = multiclass_precision(pred, label)
                recall = multiclass_recall(pred, label)
                auroc = multiclass_auroc(pred, label, num_classes=7)
                # Update metric arrays
                val_loss.append(loss.item())
                val_acc.append(acc.item())
                val_prec.append(prec.item())
                val_recall.append(recall.item())
                val_auroc.append(auroc.item())
            # Get average metrics per epoch
        avg_val_loss = sum(val_loss) / len(val_loss)
        avg_val_acc = sum(val_acc) / len(val_acc)
        avg_val_prec = sum(val_prec) / len(val_prec)
        avg_val_recall = sum(val_recall) / len(val_recall)
        avg_val_auroc = sum(val_auroc) / len(val_auroc)
        metrics['val_loss_per_epoch'].append(avg_val_loss)
        metrics['val_acc_per_epoch'].append(avg_val_acc)
        metrics['val_prec_per_epoch'].append(avg_val_prec)
        metrics['val_recall_per_epoch'].append(avg_val_recall)
        metrics['val_auroc_per_epoch'].append(avg_val_auroc)

        # If val accuracy improved, save model and set patience to 0
        # Else increase patience by 1
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), save_path)
            patience = 0
        else:
            patience += 1

        # Print status
        print(f'=== Train: loss={round(avg_train_loss, 5)}, acc={round(avg_train_acc, 5)}, prec={round(avg_train_prec, 5)}, rec={round(avg_train_recall, 5)}, auroc={round(avg_train_auroc, 5)} ==='
        + f'\n === Val: loss={round(avg_val_loss, 5)}, acc={round(avg_val_acc, 5)}, prec={round(avg_val_prec, 5)}, rec={round(avg_val_recall, 5)}, auroc={round(avg_val_auroc, 5)} ==='
        )

        # Check for early stopping
        if early_stopping is not None:
            if patience > early_stopping:
                highest_acc = max(metrics['val_acc_per_epoch'])
                highest_acc_epoch = np.argmax(metrics['val_acc_per_epoch']) + 1
                print(f'=== Early Stopping Triggered. Highest Validation Acc: {highest_acc} at epoch {highest_acc_epoch} ===')
                # Save metrics dict as json
                with open(metrics_path, 'w') as file:
                    json.dump(metrics, file)
                # End training
                break

    # Save metrics dict as json for later plotting
    with open(metrics_path, 'w') as file:
        json.dump(metrics, file)