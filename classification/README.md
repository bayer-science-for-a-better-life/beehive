# Bee Classes

Classifying single cells from the beehive into egg, larva, pollen and so on...
I used single images solely from _broodmapper.com_.

- [data/](./data/) pictures from _broodmapper.com_
- [data_bayer](./data_bayer/) segmented and labeld Bayer pictures
- [preprocessing/](./preprocessing/) pre-processing, test-partitioning, and finding transformations
- [dbs/](./dbs/) lmdbs of pre-processed and partitioned datasets
- [major_classes/](./major_classes/) I tried out a few settings. This one worked quite well. I excluded underrepresented classes from the training.
- [deploy/](./deploy/) prepare model for deployment in REST service
- [utils/](./utils/) shared code
- [label_maps.json](./label_maps.json) label maps for broodmapper dataset

> Note: This was one of my first projects using pytorch.
> There are a lot of things I unnecessarily did by hand.
> Nowadays, I would recommend using a library like [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning).

> Also: You might notice my use of [lmdb](https://pypi.org/project/lmdb/).
> The data is not that big, but my PC was really bad at that time and I did not have a cloud available.
