Reproduction Guide

This guide ensures exact reproduction of results from the Intelligent Systems for Autonomous Mining Operations paper.

System Requirements



Hardware: NVIDIA GPU with >=12GB VRAM (e.g., RTX 3080).

Software: Ubuntu 20.04, Python 3.8, CUDA 11.3.

Environment: Use environment.yml for exact dependencies.



Steps for Reproduction



Clone Repository

git clone https://github.com/ClaudioUrrea/segmentation.git

cd segmentation





Setup Environment

conda env create -f environment.yml

conda activate intelligent-mining

cd mmsegmentation

pip install -v -e .





Download Dataset and Models



Dataset: FigShare DOI: 10.6084/m9.figshare.29897300.

Place in data/ as per dataset\_preparation.md.

Pretrained weights in mmsegmentation/checkpoints/pretrained\_weights/.

Trained models: Download optimal checkpoints from FigShare.





Train Models



Use scripts/For\_training.ipynb with optimal configs:

ResNet-50: configs/ResNet/Resnet\_lr0.001\_Clahe\_photo.py

MobileNetV2: configs/MobileNet/Mobilenet\_lr0.001\_Clahe\_photo.py

SegFormer-B0: configs/SegFormer/Segformer\_lr0.0001\_Clahe\_photo.py

Twins-PCPVT-S: configs/Twins/Twins\_lr0.01\_photo.py





Set seed for reproducibility: random.seed(42), torch.manual\_seed(42).





Evaluate Models



Use scripts/For\_testing.ipynb on target domains.

Compute metrics matching Statistical\_Analysis\_Complete.xlsx (e.g., mIoU means: ResNet LS=84.58, SG=80.11).





Generate Figures and Tables



Training curves: scripts/Train\_curves.py

Boxplots: scripts/boxplot\_graphs.py

Efficiency: scripts/Size\_model.ipynb and Measure\_Speed\_Inference\_Bilateral\_Model.ipynb







Expected Results



Match paper results within ±0.5% mIoU due to hardware variations.

Generalization gaps: ResNet=15.99, etc., as in Results\_Consolidated.xlsx.



Common Issues



Determinism: Enable torch.backends.cudnn.deterministic = True.

Batch Size: Ensure GPU memory supports batch\_size=4.

Contact: claudio.urrea@usach.cl for discrepancies.



