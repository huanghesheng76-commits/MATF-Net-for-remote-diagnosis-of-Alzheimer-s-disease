import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, images_path, images_class, transform=None):
        self.root_path = images_path
        self.ap_path = os.path.join(self.root_path, "end-AP")
        self.cp_path = os.path.join(self.root_path, "end-CP")
        self.sp_path = os.path.join(self.root_path, "end-SP")
        self.images_class = images_class
        self.transform = transform

        self.ap_ad_list = os.listdir(
            os.path.join(self.ap_path, self.images_class[0])
        )
        self.ap_mci_list = os.listdir(
            os.path.join(self.ap_path, self.images_class[1])
        )
        self.ap_list = self.ap_ad_list + self.ap_mci_list

        self.cp_ad_list = os.listdir(
            os.path.join(self.cp_path, self.images_class[0])
        )
        self.cp_mci_list = os.listdir(
            os.path.join(self.cp_path, self.images_class[1])
        )

        self.sp_ad_list = os.listdir(
            os.path.join(self.sp_path, self.images_class[0])
        )
        self.sp_mci_list = os.listdir(
            os.path.join(self.sp_path, self.images_class[1])
        )

    def __len__(self):
        return (
            len(os.listdir(os.path.join(self.ap_path, self.images_class[0])))
            + len(os.listdir(os.path.join(self.ap_path, self.images_class[1])))
        )

    def __getitem__(self, item):
        image_name = self.ap_list[item]

        if image_name in self.ap_ad_list:
            label = 0
        else:
            label = 1

        image_ap = Image.open(
            os.path.join(self.ap_path, self.images_class[label], image_name)
        )
        image_cp = Image.open(
            os.path.join(self.cp_path, self.images_class[label], image_name)
        )
        image_sp = Image.open(
            os.path.join(self.sp_path, self.images_class[label], image_name)
        )

        if self.transform is not None:
            image_ap = self.transform(image_ap)
            image_cp = self.transform(image_cp)
            image_sp = self.transform(image_sp)

        assert image_ap.ndim == 3, "image_ap should be a 3D tensor after transform"
        assert image_cp.ndim == 3, "image_cp should be a 3D tensor after transform"
        assert image_sp.ndim == 3, "image_sp should be a 3D tensor after transform"

        return [image_ap, image_cp, image_sp], label

    @staticmethod
    def collate_fn(batch):
        batch_images, batch_labels = zip(*batch)

        images_ap = torch.stack(
            [torch.stack(patches, dim=0) for patches in batch_images[0]],
            dim=0
        )
        images_cp = torch.stack(
            [torch.stack(patches, dim=0) for patches in batch_images[1]],
            dim=0
        )
        images_sp = torch.stack(
            [torch.stack(patches, dim=0) for patches in batch_images[2]],
            dim=0
        )

        labels = torch.tensor(batch_labels, dtype=torch.long)

        print("images_ap:", type(images_ap[0]))
        print("images_ap:", images_ap[0].shape[0])

        return [images_ap, images_cp, images_sp], labels