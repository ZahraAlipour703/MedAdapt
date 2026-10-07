from datasets.breakhis import (
    BreakHisDataset,
    get_default_transform
)


dataset = BreakHisDataset(
    root_dir="data/BreakHis",
    transform=get_default_transform()
)


print("Dataset size:", len(dataset))


sample = dataset[0]

print(sample["image"].shape)
print(sample["label"])
print(sample["path"])