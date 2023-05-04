from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import glob
from PIL import Image
import numpy as np
import torch


def mask2color(masks):
    COLOR_MAP = {
        0: [0, 0, 0],
        1: [204, 0, 0],
        2: [76, 153, 0],
        3: [204, 204, 0],
        4: [51, 51, 255],
        5: [204, 0, 204],
        6: [0, 255, 255],
        7: [255, 204, 204],
        8: [102, 51, 0],
        9: [255, 0, 0],
        10: [102, 204, 0],
        11: [255, 255, 0],
        12: [0, 0, 153],
        13: [0, 0, 204],
        14: [255, 51, 153],
        15: [0, 204, 204],
        16: [0, 51, 0],
        17: [255, 153, 51],
        18: [0, 204, 0]}

    masks = torch.argmax(masks, dim=1).float()
    sample_mask = torch.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 3), dtype=torch.float)
    for key in COLOR_MAP.keys():
        sample_mask[masks == key] = torch.tensor(COLOR_MAP[key], dtype=torch.float)
    sample_mask = sample_mask.permute(0, 3, 1, 2)
    return sample_mask


class CelebA(Dataset):
    """CelelebA Dataset"""

    def __init__(self, image_path, parsing_path, output_size, **kwargs):
        super().__init__()

        self.data = glob.glob(image_path)
        self.parsing = glob.glob(parsing_path)
        self.data = sorted(self.data)
        self.parsing = sorted(self.parsing)
        self.output_size = output_size
        assert len(self.data) > 0 and len(self.parsing) > 0         # "Can't find data; make sure you specify the path to your dataset"

        self.transform_image = transforms.Compose(
            [transforms.Resize(576),
             transforms.CenterCrop(512),
             transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5]),
             transforms.Resize((output_size, output_size), interpolation=InterpolationMode.BICUBIC)
        ])

        self.transform_parsing = transforms.Compose(
            [transforms.Resize(576, interpolation=InterpolationMode.NEAREST),
             transforms.CenterCrop(512),
             transforms.ToTensor(),
             transforms.Resize((output_size, output_size), interpolation=InterpolationMode.NEAREST)
             ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = Image.open(self.data[index]).convert('RGB')
        parsing = Image.open(self.parsing[index]).convert('RGB')
        p = np.random.random(1)
        X = self.transform_image(X)
        # X = transforms.Resize((self.output_size, self.output_size), interpolation=InterpolationMode.BICUBIC)(X)
        parsing = self.transform_parsing(parsing)
        # Y = transforms.Resize((64, 64), interpolation=InterpolationMode.BICUBIC)(Y)
        if p > 0.5:
            X = transforms.RandomHorizontalFlip(1)(X)
            parsing = transforms.RandomHorizontalFlip(1)(parsing)

        parsing = self._mask_labels((parsing * 255.)[0])
        parsing = (parsing - 0.5) / 0.5

        return X, parsing

    def _mask_labels(self, mask_np):
        label_size = 19
        # print(mask_np)
        labels = np.zeros((label_size, mask_np.shape[0], mask_np.shape[1]), dtype=np.float32)
        for i in range(label_size):
            # print(mask_np)
            labels[i][mask_np == i] = 1.0
        # labels[0] = 0.  # set background class to zero
        return labels


def get_dataset(name,  **kwargs):
    dataset = globals()[name](**kwargs)
    return dataset
