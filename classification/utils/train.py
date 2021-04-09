import torch
from torch.utils.data import DataLoader
from utils.training_helper import TrainingHelper
from utils.machine import device, workers


def convert_target(y, criterion):
    name = str(criterion)
    if "CrossEntropyLoss" in name:
        return y.long()
    if "BCEWithLogitsLoss" in name:
        return torch.nn.functional.one_hot(y).float()
    return y


def train_with_params(
    ds_train,
    ds_val,
    model,
    patience,
    batch_size,
    num_epochs,
    scheduler,
    optimizer,
    criterion,
    progress_file,
    model_file,
    samplerfact=None,
    state_dict=None,
):
    """
    ds_train: torch Dataset instance for training data
    ds_val: torch Dataset instance for validation data
    model: torch.nn model instance
    patience: int how many epochs before early stopping
    batch_size: int batch size
    num_epochs: int for max num epochs (if no early stopping)
    scheduler: learning rate scheduler instance from torch.optim.lr_scheduler
    optimizer: optim optimizer instance
    criterion: loss function instance
    progress_file: path to file where every epoch train/val is recorded
    model_file: path to file where best model is saved
    samplerfact: a sampler factory that has a get_sampler() which takes the dataset labels
    state_dict: str of path to state dict in case model was pre-trained
    """
    if state_dict is not None:
        model.load_state_dict(torch.load(state_dict, map_location=device))
    model.to(device)

    # data loading
    if samplerfact is not None:
        sampler_train = samplerfact(ds_train.labels)
        sampler_val = samplerfact(ds_val.labels)
        dataloaders = dict(
            train=DataLoader(
                ds_train,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                sampler=sampler_train,
            ),
            val=DataLoader(
                ds_val,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                sampler=sampler_val,
            ),
        )
    else:
        dataloaders = dict(
            train=DataLoader(
                ds_train, batch_size=batch_size, shuffle=True, num_workers=workers
            ),
            val=DataLoader(
                ds_val, batch_size=batch_size, shuffle=True, num_workers=workers
            ),
        )

    dataset_sizes = dict(train=len(ds_train), val=len(ds_val))

    # training
    helper = TrainingHelper(
        progress_file=progress_file,
        model_file=model_file,
        patience=patience,
        best_stat=0,
        stat_name="Acc",
    )

    helper.start(model.state_dict())

    for epoch in range(num_epochs):
        epoch_progress = dict()

        for phase in ["train", "val"]:
            if phase == "train":
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for x, y in dataloaders[phase]:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == "train"):
                    out = model(x)
                    _, preds = torch.max(out, 1)
                    loss = criterion(out, convert_target(y, criterion))

                    # backward
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * len(y)
                running_corrects += (preds == y).sum().item()

            # get progress
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_progress[phase] = [epoch_loss, epoch_acc]

            # early stopping
            if phase == "val":
                try:
                    helper.early_stopping(
                        model_dict=model.state_dict(), stat=epoch_acc, better_if="gt"
                    )
                except AttributeError:
                    return helper.best_stat

        # append progress
        helper.epoch_end(epoch, epoch_progress)

    # done
    helper.done()
    return helper.best_stat
