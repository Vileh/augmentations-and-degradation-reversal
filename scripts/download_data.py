from wildlife_datasets import datasets

wildlife_datasets = [
    "OpenCows2020",
    "ATRW",
    "BelugaID",
    "CTai",
    "GiraffeZebraID",
    "Giraffes",
    "HumpbackWhaleID",
    "HyenaID2022",
    "IPanda50",
    "LeopardID2022",
    "MacaqueFaces",
    "NyalaData",
    # "SealID", # Requires URL
    "SeaTurtleIDHeads",
    # "StripeSpotter", # Downloaded externally
    "WhaleSharkID",
    "ZindiTurtleRecall"
]

for dataset in wildlife_datasets:
    dataset_func = getattr(datasets, dataset)

    dataset_func.get_data(f'../datasets/{dataset}')