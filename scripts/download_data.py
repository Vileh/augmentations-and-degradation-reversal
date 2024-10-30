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
    "SealID",
    "SeaTurtleIDHeads",
    # "StripeSpotter", # Downloaded externally
    "WhaleSharkID",
    "ZindiTurtleRecall"
]

for dataset in wildlife_datasets:
    dataset_func = getattr(datasets, dataset)

    if dataset == "SealID":
        dataset_func.get_data(f'../datasets/{dataset}', url="https://download.fairdata.fi:443/download?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3MzAyMDM2MzMsImRhdGFzZXQiOiIyMmI1MTkxZS1mMjRiLTQ0NTctOTNkMy05NTc5N2M5MDBmYzAiLCJwYWNrYWdlIjoiMjJiNTE5MWUtZjI0Yi00NDU3LTkzZDMtOTU3OTdjOTAwZmMwXzVxcmNrZ290LnppcCIsImdlbmVyYXRlZF9ieSI6ImE0MzdhOGRmLTllZjEtNDVjOC1hYWQxLTQ4MmYxYTA4ZTM1YiIsInJhbmRvbV9zYWx0IjoiMjhjYjE1MzYifQ.HzPaBTEoa8Scw5f9pdGicMFJXvsvkvdGg8r_y3fRawo")
    else:
        dataset_func.get_data(f'../datasets/{dataset}')