def train(epoch, model, train_loader, criterion, optimizer):
    model.train() # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    device = next(model.parameters()).device

    for batch_idx, (inputs, targets) in enumerate(train_loader):
 
        if isinstance(inputs, (list, tuple)):
            inputs = [i.to(device) for i in inputs]
        else:
            inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    print(f'Epoch {epoch}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    return epoch_loss, epoch_acc
