"""
Transformations that I found suitable for these images.
Derived from `preprocessing/`
"""
import torchvision.transforms as trans

resize = trans.Resize((150, 150))
resize_small = trans.Resize((50, 50))
totensor = trans.ToTensor()

augment = trans.Compose(
    [
        trans.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01),
        trans.RandomAffine(degrees=0.05, scale=(0.9, 1.1), shear=2),
        trans.RandomResizedCrop((150, 150), scale=(0.9, 1), ratio=(1, 1)),
        trans.RandomVerticalFlip(),
        trans.RandomHorizontalFlip(),
    ]
)

augment_small = trans.Compose(
    [
        trans.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        trans.RandomAffine(degrees=0.05, scale=(0.9, 1.1), shear=2),
        trans.RandomVerticalFlip(),
        trans.RandomHorizontalFlip(),
        resize_small,
    ]
)

augment_nohue = trans.Compose(
    [
        trans.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        trans.RandomAffine(degrees=0.05, scale=(0.9, 1.1), shear=2),
        trans.RandomResizedCrop((150, 150), scale=(0.9, 1), ratio=(1, 1)),
        trans.RandomVerticalFlip(),
        trans.RandomHorizontalFlip(),
    ]
)

augment_nocolor = trans.Compose(
    [
        trans.RandomAffine(degrees=0.05, scale=(0.9, 1.1), shear=2),
        trans.RandomResizedCrop((150, 150), scale=(0.9, 1), ratio=(1, 1)),
        trans.RandomVerticalFlip(),
        trans.RandomHorizontalFlip(),
    ]
)
