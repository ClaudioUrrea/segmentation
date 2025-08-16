**Intelligent Systems for Autonomous Mining Operations: Real-Time Robust Road Segmentation**



This repository implements the intelligent framework for robust road segmentation in autonomous mining vehicle systems, as described in the paper:

Intelligent Systems for Autonomous Mining Operations: Real-Time Robust Road Segmentation Using Single-Domain Generalization to Address Sensor-Based Visual Degradations

Authors: Claudio Urrea\*, Maximiliano Vélez

Affiliation: Electrical Engineering Department, University of Santiago of Chile

Correspondence: claudio.urrea@usach.cl

Abstract

Background: Intelligent autonomous systems in open-pit mining operations face critical challenges in perception and decision-making due to sensor-based visual degradations, particularly lens soiling and sun glare, which significantly compromise the performance and safety of integrated mining automation systems.

Methods: We propose a comprehensive intelligent framework leveraging single-domain generalization with traditional data augmentation techniques, specifically Photometric Distortion (PD) and Contrast Limited Adaptive Histogram Equalization (CLAHE), integrated within the BiSeNetV1 architecture. Our systematic approach evaluated four state-of-the-art backbones: ResNet-50, MobileNetV2 (Convolutional Neural Networks (CNN)-based), SegFormer-B0, and Twins-PCPVT-S (ViT-based) within an end-to-end autonomous system architecture. The model was trained on clean images from the AutoMine dataset and tested on degraded visual conditions without requiring architectural modifications or additional training data from target domains.

Results: ResNet-50 demonstrated superior system robustness with mean Intersection over Union (IoU) of 84.58% for lens soiling and 80.11% for sun glare scenarios. MobileNetV2 achieved optimal computational efficiency for real-time autonomous systems with 109.58 Frames Per Second (FPS) inference speed while maintaining competitive accuracy (81.54% and 71.65% mIoU respectively). Vision Transformers showed superior stability in system performance but lower overall performance under severe degradations.

Conclusions: The proposed intelligent augmentation-based approach maintains high accuracy while preserving real-time computational efficiency, making it suitable for deployment in autonomous mining vehicle systems. The methodology provides a practical solution for robust perception systems without requiring expensive multi-domain training datasets.

Keywords: intelligent autonomous systems; mining automation systems; real-time perception systems; domain generalization; visual degradation robustness; embedded intelligent systems; autonomous vehicle systems; intelligent mining operations; sensor-based decision making

Key Features



Single-domain generalization for handling visual degradations (lens soiling, sun glare) in mining environments.

Implementation based on MMSegmentation framework with custom BiSeNetV1 architecture.

Evaluation of multiple backbones: ResNet-50, MobileNetV2, SegFormer-B0, Twins-PCPVT-S.

Data augmentation techniques: Photometric Distortion (PD) and CLAHE.

Real-time inference capabilities for embedded systems.

Comprehensive documentation, scripts for training/evaluation, and reproducible experiments.



Installation

Follow the detailed instructions in docs/installation.md. Summary:



Clone the repository:

git clone https://github.com/ClaudioUrrea/segmentation.git

cd segmentation





Create conda environment:

conda env create -f environment.yml

conda activate intelligent-mining





Install MMSegmentation:

cd mmsegmentation

pip install -v -e .







Dataset

Download the AutoMine dataset from FigShare: https://doi.org/10.6084/m9.figshare.29897300



Source domain: Clean images for training.

Target domains: Lens soiling and sun glare for testing.



See docs/dataset\_preparation.md for setup instructions.

Usage

Training

Use the Jupyter notebook:

jupyter notebook scripts/For\_training.ipynb



Select a config file from mmsegmentation/configs/ (e.g., ResNet/Resnet\_lr0.001\_Clahe\_photo.py for optimal ResNet-50).

Testing

For evaluation on target domains:

jupyter notebook scripts/For\_testing.ipynb



Inference

For single image inference:

jupyter notebook examples/single\_image\_inference.ipynb



Results and Reproduction

Raw experimental results, trained models, and figures are available on FigShare: https://doi.org/10.6084/m9.figshare.29897300.

For reproduction, follow docs/reproduction\_guide.md.

Contributing

See CONTRIBUTING.md for guidelines.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Citation

If you use this code or dataset in your research, please cite the paper:

@article{urrea2025intelligent,

&nbsp; title={Intelligent Systems for Autonomous Mining Operations: Real-Time Robust Road Segmentation},

&nbsp; author={Urrea, Claudio and V{\\'e}lez, Maximiliano},

&nbsp; journal={Systems},

&nbsp; volume={13},

&nbsp; year={2025},

&nbsp; publisher={MDPI},

&nbsp; doi={10.3390/xxxxx}

}



Acknowledgments



Based on MMSegmentation.

Dataset derived from AutoMine with custom annotations.



For questions, contact claudio.urrea@usach.cl.

