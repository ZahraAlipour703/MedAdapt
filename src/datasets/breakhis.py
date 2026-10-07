import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


class BreakHisDataset(Dataset):
    """
    BreakHis Breast Histopathology Dataset

    Binary classification:
        0 -> benign
        1 -> malignant
    """

    def __init__(
        self,
        root_dir,
        transform=None
    ):

        self.root_dir = root_dir
        self.transform = transform

        self.images = []
        self.labels = []

        self._load_dataset()


    def _load_dataset(self):

        classes = {
            "benign": 0,
            "malignant": 1
        }


        for class_name, label in classes.items():

            class_path = os.path.join(
                self.root_dir,
                class_name
            )


            if not os.path.exists(class_path):
                continue


            for root, _, files in os.walk(class_path):

                for file in files:

                    if file.lower().endswith(
                        (".png", ".jpg", ".jpeg")
                    ):

                        image_path = os.path.join(
                            root,
                            file
                        )

                        self.images.append(image_path)
                        self.labels.append(label)


    def __len__(self):

        return len(self.images)


    def __getitem__(self, index):

        image_path = self.images[index]
        label = self.labels[index]


        image = Image.open(
            image_path
        ).convert("RGB")


        if self.transform:
            image = self.transform(image)


        return {
            "image": image,
            "label": label,
            "path": image_path
        }



def get_default_transform():

    return transforms.Compose(
        [

            transforms.Resize(
                (224,224)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],

                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )

        ]
    )