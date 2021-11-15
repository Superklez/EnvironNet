import time
import copy
import torch
import torch.nn.functional as F

def train_model(
    model: object,
    dataloaders: dict,
    optimizer: object,
    criterion: object,
    epochs: int,
    training_state: dict = None,
    device: str = "cpu"
):

    training = model.training
    
    start_epoch = training_state["start_epoch"]
    losses = training_state["losses"]
    accuracies = training_state["accuracies"]
    best_acc = training_state["best_acc"]
    best_model = training_state["best_model"]
    smallest_loss = training_state["smallest_loss"]
    smallest_model = training_state["smallest_model"]

    start_time = time.time()
    for epoch in range(start_epoch, start_epoch+epochs):
        print(f"Epoch {epoch}/{start_epoch+epochs-1}")
        print('-' * 15)

        for phase in dataloaders.keys():
            if phase == "train":
                model.train()

            elif phase == "val":
                model.eval()

            running_loss = 0
            running_corrects = 0

            for b, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = F.softmax(outputs, 1).argmax(dim=1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc)

            print("{} Loss: {:.4f}, Accuracy: {:.2f}%".format(
                phase.capitalize(), epoch_loss, epoch_acc * 100))
            
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())

            if phase == "val" and epoch_loss < smallest_loss:
                smallest_loss = epoch_loss
                smallest_model = copy.deepcopy(model.state_dict())

        print()
    
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "last_epoch": epoch,
        "losses": losses,
        "accuracies": accuracies,
        "best_acc": best_acc,
        "best_model": best_model,
        "smallest_loss": smallest_loss,
        "smallest_model": smallest_model
    }

    time_elapsed = time.time() - start_time
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60,
        time_elapsed % 60))

    if training is False:
        model.eval()

    return model, state

def test_model(
    model: object,
    test_loader: object,
    optimizer: object,
    criterion: object,
    device: str = "cpu",
    pin_memory: bool = False,
):

    training = model.training

    start_time = time.time()

    running_loss = 0
    running_corrects = 0

    all_preds = []
    all_labels = []

    for b, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device, non_blocking=pin_memory)
        labels = labels.to(device, non_blocking=pin_memory)
        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = F.softmax(outputs, 1).argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.data.tolist())

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects / len(test_loader.dataset)

    time_elapsed = time.time() - start_time
    print("Testing complete in {:.0f}m {:.0f}s".format(time_elapsed // 60,
        time_elapsed % 60))

    if training is False:
        model.eval()

    return model, {"loss": test_loss, "accuracy": test_acc,
        "predictions": all_preds, "labels": all_labels}