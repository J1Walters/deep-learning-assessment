import torch
from torcheval.metrics.functional import multiclass_accuracy, multiclass_precision, multiclass_recall, multiclass_auroc
import json
from time import time
from tqdm import tqdm
from loss import WeightedCELoss

def test(model, test_data, metrics_path='./test_metrics.json'):
    # Check if can run on cuda
    if torch.cuda.is_available():
        device = 'cuda'
        model.cuda()
    else:
        device = 'cpu'
    # Tell user what device running on
    print(f'Running on {device}')
    # Predictions
    y_true = []
    y_pred = []
    # Metric arrays
    test_loss = []
    test_acc = []
    test_prec = []
    test_recall = []
    test_auroc = []
    # Get loss function
    loss_func = WeightedCELoss()
    # Get start time
    start_time = time()
    # Put model into eval mode
    model.eval()
    for image, label in tqdm(test_data):
        # Move to device
        image, label = image.to(device), label.to(device)
        # Don't calculate gradient
        with torch.no_grad():
            # Make predictions
            pred = model(image)
            # Calculate loss
            loss = loss_func(pred, label)
            # Append truth labels and predictions
            y_true.append(label)
            y_pred.append(torch.argmax(pred, 1))
            # Calculate metrics
            acc = multiclass_accuracy(pred, label)
            prec = multiclass_precision(pred, label)
            recall = multiclass_recall(pred, label)
            auroc = multiclass_auroc(pred, label, num_classes=7)
            # Update metric arrays
            test_loss.append(loss.item())
            test_acc.append(acc.item())
            test_prec.append(prec.item())
            test_recall.append(recall.item())
            test_auroc.append(auroc.item())
    # Get end time
    end_time = time()
    # Calculate time taken
    time_taken = end_time - start_time
    # Get average metrics over full batch
    avg_test_loss = sum(test_loss) / len(test_loss)
    avg_test_acc = sum(test_acc) / len(test_acc)
    avg_test_prec = sum(test_prec) / len(test_prec)
    avg_test_recall = sum(test_recall) / len(test_recall)
    avg_test_auroc = sum(test_auroc) / len(test_auroc)

    # Get metrics dict
    metrics = {
        'test_loss':avg_test_loss,
        'test_acc':avg_test_acc,
        'test_prec':avg_test_prec,
        'test_recall':avg_test_recall,
        'test_auroc':avg_test_auroc
    }

    print(f'=== Time Taken: {time_taken} ===')
    print(metrics)

    # Save metrics as json
    with open(metrics_path, 'w') as file:
        json.dump(metrics, file)

    # Return ground truth and predictions
    return y_true, y_pred