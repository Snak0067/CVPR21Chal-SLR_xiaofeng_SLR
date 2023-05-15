import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def train_epoch(model, criterion, optimizer, dataloader, device, epoch, logger, log_interval, writer):
    model.train()
    losses = []
    all_label = []
    all_pred = []

    for batch_idx, data in enumerate(tqdm(dataloader)):

        # get the inputs and labels
        inputs, labels = data['data'].to(device), data['label'].to(device)

        optimizer.zero_grad()
        # forward
        outputs = model(inputs)
        if isinstance(outputs, list):
            outputs = outputs[0]
        # compute the loss
        loss = criterion(outputs, labels.squeeze())
        losses.append(loss.item())

        # compute the accuracy
        prediction = torch.max(outputs, 1)[1]
        all_label.extend(labels.squeeze().cpu())
        all_pred.extend(prediction.cpu())
        score = accuracy_score(labels.squeeze().cpu().data.numpy(), prediction.cpu().data.numpy())

        # backward & optimize
        loss.backward()
        optimizer.step()

        logger.info("epoch {:3d} | iteration {:5d} | Loss {:.6f} | Acc {:.2f}%".format(epoch + 1, batch_idx + 1,
                                                                                       loss.item(), score * 100))
    print("Compute the average loss & accuracy")
    training_loss = sum(losses) / len(losses)
    all_label = torch.stack(all_label, dim=0).to(device)
    all_pred = torch.stack(all_pred, dim=0).to(device)
    training_acc = accuracy_score(all_label.squeeze().cpu().data.numpy(), all_pred.cpu().data.numpy())

    # Log
    # writer.add_scalars('Loss', {'train': training_loss}, epoch + 1)
    # writer.add_scalars('Accuracy', {'train': training_acc}, epoch + 1)
    logger.info(
        "Average Training Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch + 1, training_loss, training_acc * 100))
