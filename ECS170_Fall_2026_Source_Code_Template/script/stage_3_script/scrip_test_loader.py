from local_code.stage_3_code.Dataset_loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import Method_CNN
import torch

# (file name, single_channel?)
configs = [("CIFAR", False)]

for name, single in configs:
    loader = Dataset_Loader(name, '')
    loader.dataset_source_folder_path = 'data/stage_3_data/'
    loader.dataset_source_file_name = name
    d = loader.load(single_channel=single)

    input_channels = d['train']['X'].shape[1]
    num_classes = len(set(d['train']['y']))

    model = Method_CNN(
        "CNN",
        name + " CNN",
        input_channels = input_channels,
        num_classes = num_classes,
    )

    model.data = d

    results = model.run()

    accuracy = (
        results['pred_y'].numpy() == results["true_y"]
    ).mean()

    print("Test accuracy:", accuracy)

    print(name, "Predictions:", results['pred_y'].shape)

    print(name,
          "| train X:", d['train']['X'].shape,
          "| test X:", d['test']['X'].shape,
          "| labels:", d['train']['y'].min(), "-", d['train']['y'].max(),
          "| X range:", d['train']['X'].min(), "-", d['train']['X'].max())