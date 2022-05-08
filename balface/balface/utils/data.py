import torch
from torch.utils.data import DataLoader, DistributedSampler



class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


class SSLDataloader:
    '''
    only for ssl settings
    '''
    def __init__(self, datasets, samples_per_gpu, workers_per_gpu, shuffle=True, drop_last=True, dist=True):
        assert len(datasets) == 2
        assert len(samples_per_gpu) == 2 and len(workers_per_gpu) == 2
        labeled_dataset, unlabeled_dataset = datasets
        assert dist

        labeled_sampler = DistributedSampler(labeled_dataset, shuffle=shuffle)
        self.labeled_dataloader = DataLoader(labeled_dataset,
                                             sampler=labeled_sampler,
                                             batch_size=samples_per_gpu[0], num_workers=workers_per_gpu[0],
                                             drop_last=drop_last)

        unlabeled_sampler = DistributedSampler(unlabeled_dataset, shuffle=shuffle)
        self.unlabeled_dataloader = DataLoader(unlabeled_dataset,
                                               sampler=unlabeled_sampler,
                                               batch_size=samples_per_gpu[1],
                                               num_workers=workers_per_gpu[1],
                                               drop_last=drop_last)

        self.min_dataset_index = 0 if len(datasets[0])//samples_per_gpu[0] < len(datasets[1])//samples_per_gpu[1] else 1
        assert self.min_dataset_index == 0

        self.labeled_iter = iter(self.labeled_dataloader)
        self.unlabeled_iter = iter(self.unlabeled_dataloader)

    def __len__(self):
        return len(self.unlabeled_dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            labeled_data = next(self.labeled_iter)
        except:
            self.labeled_iter = iter(self.labeled_dataloader)
            labeled_data = next(self.labeled_iter)

        unlabeled_data = next(self.unlabeled_iter)

        return labeled_data, unlabeled_data