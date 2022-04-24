# 5LSM0_Cityscapes_challenge
Project repository for final assignment of 5LSM0

# Instrution to use our automatically training tool

Train and save data for one configuration file:
`python main.py --path {path_to_ configuration_file}`

Train multiple configuration file:
`python run_config.py --config config_folder_1 [config_folder_2 ...]`

please make sure the package folder scripts located with main.py file

## Configuration file template

```
{
    "architecture": {
        "prefix": "scripts.models.pos_unet.pos_unet", // the package which contains the following model
        "model": "UNet_Pos", //model class name
        "param": {  //model paramters
            "n_channels": 3,
            "n_classes": 30,
            "in_height": 128,
            "in_width": 256
        }
    },
    "param": { // training parameters
        "learning_rate": 0.0002,
        "epochs": 15,
        "batch_size": 3,
        "aug_mode": "Both", // "None": no augmentaion, "Only":no original data ,"Both" : both augmentation data nad original data
        "save_inter_model": false // true: save models for every epoch, false: only save the untrained and trained model
    },
    "augmentation": { // Augmentations will be used
        "transformation": { //Similar structure in the architecture
            "prefix": "scripts.data_augmentation",
            "transforms": [
                {
                    "name": "crop.RandomCropAndResize", // python file name and class name
                    "param": { //tranformation paramters
                        "pr": 1, "width_lim": [ 30, 250 ], "height_lim": [ 30, 120 ] }
                },
                { "name": "flip.RandomHorizontalFlip2",
                    "param": { "pr": 1 }
                },
                {
                    "name": "pixels_change.RandomRgb2Gray2",
                    "param": { "pr": 1 }
                },
                {
                    "name": "rotate.RandomRotation",
                    "param": { "pr": 1 }
                }
            ]
        }
    },
    "path": {
        "ROOT_PATH": "saves", // the root path where the results will be saved to
        "SAVE_PATH": "UNet_Pos-lr-0p0002-AUGMODE_Both-EXP_1", // the folder where the result will be saved to
        "MODEL_PATH": "models", // the folder name where the trained models are saved
        "CHECK_POINT": "", // Not used currently
        "MODEL_PREFIX": "UNet_Pos", // the save prefix of models
        "RESULT_PATH": "res" // the folder name where the results and records are saved
    }
}
```


## Folder structure
```
.
├── Images
├── config // where the configuration files are stored
│   ├── unet
│   ├── pos_unet
│   ├── res_unet
│   ├── r2unet
│   ├── unet_1
│   ├── unet_2
│   └── unet_2_1
├── scripts //our implementations, will be shown bellow with more detail
│   ├── data_augmentation
│   ├── dataset
│   ├── models
│   │   ├── pos_unet //pos-unet
│   │   ├── r2unet
│   │   ├── res_unet
│   │   ├── cat_unet
│   │   └── unet //baseline
│   └── utilities
├── analysis // notebooks that was used to draw images. Just an example. Not for automated using
├── data
├── main.py //main training script
├── conf.py 
├── run_config.py
└── test
```
Detail structure for scripts
```
scripts
├── Trainer.py
├── data_augmentation
│   ├── crop.py
│   ├── flip.py
│   ├── invert.py
│   ├── pixels_change.py
│   └── rotate.py
├── dataset
│   └── dataset.py
├── metrics.py
├── models
│   ├── cat_unet
│   │   ├── __init__.py
│   │   ├── unet_model.py
│   │   └── unet_parts.py
│   ├── pos_unet
│   │   ├── PositionalEncoding.py
│   │   ├── pos_unet.py
│   │   └── unet_parts.py
│   ├── r2unet
│   │   ├── __init__.py
│   │   ├── network.py
│   │   ├── r2unet.py
│   │   └── rrcnblock.py
│   ├── res_unet
│   │   ├── res_unet.py
│   │   └── unet_parts.py
│   └── unet
│       ├── __init__.py
│       ├── unet_model.py
│       └── unet_parts.py
└── utilities
    ├── draw.py
    ├── parse_config.py
    ├── test_segmentation.py
    └── visualize_fmap.py
    ```