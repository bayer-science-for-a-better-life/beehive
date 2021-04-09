import time
import torch
import csv


class TrainingHelper(object):
    def __init__(self, progress_file, model_file, patience, best_stat, stat_name):
        self.model_file = model_file
        self.progress_file = progress_file
        self.patience = patience
        self.best_stat = best_stat
        self.stat_name = stat_name

    def start(self, model_dict):
        torch.save(model_dict, self.model_file)
        with open(self.progress_file, "w") as ouf:
            writer = csv.writer(ouf)
            writer.writerow(
                (
                    "Epoch",
                    "Train-Loss",
                    "Train-%s" % self.stat_name,
                    "Val-Loss",
                    "Val-%s" % self.stat_name,
                )
            )
        self.since = time.time()
        self.waiting = 0

    def best_model(self, model):
        torch.save(model.state_dict(), self.model_file)
        self.waiting = 0

    def early_stopping(self, model_dict, stat, better_if):
        if better_if == "gt":
            isbetter = stat > self.best_stat
        elif better_if == "lt":
            isbetter = stat < self.best_stat
        else:
            raise ValueError("better_if must be gt or lt")

        if isbetter:
            torch.save(model_dict, self.model_file)
            self.waiting = 0
            self.best_stat = stat
        else:
            self.waiting += 1

        if self.waiting >= self.patience:
            time_elapsed = time.time() - self.since
            print(
                "Training complete in {:.0f}m {:.0f}s".format(
                    time_elapsed // 60, time_elapsed % 60
                )
            )
            print("Best Val {}: {:4f}".format(self.stat_name, self.best_stat))
            raise AttributeError("Early Stopping")

    def epoch_end(self, epoch, progress):
        row = [epoch + 1] + progress["train"] + progress["val"]
        with open(self.progress_file, "a") as ouf:
            writer = csv.writer(ouf)
            writer.writerow(row)

        if epoch % 10 == 0:
            print(
                "{} Train-Loss: {:.4f} Train-{}: {:.4f} Val-Loss: {:.4f} Val-{}: {:.4f}".format(
                    row[0],
                    row[1],
                    self.stat_name,
                    row[2],
                    row[3],
                    self.stat_name,
                    row[4],
                )
            )

    def done(self):
        time_elapsed = time.time() - self.since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best Val {}: {:4f}".format(self.stat_name, self.best_stat))
