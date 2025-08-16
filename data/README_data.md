AutoMine Dataset Description

This directory contains the AutoMine dataset for intelligent mining systems research.

Structure

data/

├── automine1d/                     # Source domain (clean images)

│   ├── images/

│   │   ├── IMG\_001.jpg

│   │   ├── IMG\_002.jpg

│   │   └── ... (100 images)

│   ├── annotations/

│   │   ├── IMG\_001.png

│   │   ├── IMG\_002.png

│   │   └── ... (100 masks)

│   └── metadata/

│       ├── train\_split.txt

│       ├── val\_split.txt

│       └── image\_info.csv

├── automine1d\_distortion/          # Target domains

│   ├── lens\_soiling/

│   │   ├── images/

│   │   │   ├── original/           # 10 original images

│   │   │   └── augmented/          # 90 augmented images

│   │   ├── annotations/

│   │   │   ├── original/           # 10 original masks

│   │   │   └── augmented/          # 90 augmented masks

│   │   └── metadata/

│   │       └── test\_split.txt

│   └── sun\_glare/

│       ├── images/

│       │   ├── original/           # 10 original images

│       │   └── augmented/          # 90 augmented images

│       ├── annotations/

│       │   ├── original/           # 10 original masks

│       │   └── augmented/          # 90 augmented masks

│       └── metadata/

│           └── test\_split.txt

└── metadata/

&nbsp;   ├── dataset\_statistics.csv

&nbsp;   └── augmentation\_parameters.yaml



Setup

Download the dataset from FigShare: https://doi.org/10.6084/m9.figshare.29897300.

Follow docs/dataset\_preparation.md for detailed instructions.

