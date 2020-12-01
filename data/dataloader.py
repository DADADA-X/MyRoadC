from base import BaseDataLoader


class DataLoader(BaseDataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, validation_split, num_workers)
