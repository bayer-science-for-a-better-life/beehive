from torch.utils.data import Dataset
from torchvision.transforms import Compose
from utils.dbconnector import DBConnector


class BeeDataset(Dataset):
    def __init__(self, db_uri, subdb, transforms):
        """
        db_uri: connection str for lmdb
        subdb: 'train' or 'val'
        transforms: list of transformations for imgs before returning
        """
        db = DBConnector(db_uri)
        self.transform = Compose(transforms)
        db.open()
        self.imgs = db.get_all_features(subdb)
        self.labels = db.get_all_targets(subdb)
        db.close()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.transform(self.imgs[idx])
        y = self.labels[idx]
        return x, y
