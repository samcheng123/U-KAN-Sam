import os

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
    
    # Load the image
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        if img is None:
            raise FileNotFoundError(f"Image file not found: {os.path.join(self.img_dir, img_id + self.img_ext)}")
    
    # Load the mask(s)
        mask = []
        for i in range(self.num_classes):
            mask_path = os.path.join(self.mask_dir, str(i), img_id + self.mask_ext)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
            if mask_img is None:
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
            mask.append(mask_img[..., None])  # Expand dimensions to ensure shape consistency
    
        mask = np.dstack(mask)  # Stack masks along the third dimension

    # Apply transformations, if any
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
    
    # Normalize and rearrange dimensions for PyTorch (channels-first)
        img = img.astype('float32') / 255.0
        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)  # Convert to torch tensor
    
        mask = mask.astype('float32') / 255.0
        mask = torch.tensor(mask.transpose(2, 0, 1), dtype=torch.float32)  # Convert to torch tensor

    # Ensure binary mask values
        mask[mask > 0] = 1.0

        return img, mask, {'img_id': img_id}

