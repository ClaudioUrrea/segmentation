Evaluation Guide

This guide details how to evaluate models in the Intelligent Mining Systems project on target domains (lens soiling, sun glare).

Prerequisites



Trained model checkpoints in mmsegmentation/checkpoints/trained\_models/.

Target datasets in data/automine1d\_distortion/.

MMSegmentation installed as per mmsegmentation\_setup.md.



Evaluation Steps



Select Configuration and CheckpointExample: ResNet-50 optimal

config\_file = 'mmsegmentation/configs/ResNet/Resnet\_lr0.001\_Clahe\_photo.py'

checkpoint\_file = 'mmsegmentation/checkpoints/trained\_models/ResNet50\_optimal\_checkpoint.pth'





Run Evaluation ScriptUse scripts/For\_testing.ipynb:

jupyter notebook scripts/For\_testing.ipynb





Load model with init\_segmentor(config\_file, checkpoint\_file, device='cuda:0').

Evaluate on lens soiling and sun glare datasets.

Compute mIoU using mmseg.apis.multi\_gpu\_test or custom metrics in utils/evaluation\_metrics.py.





Metrics



Primary: mIoU (Mean Intersection over Union).

Secondary: FPS (Frames Per Second), model size.

Use utils/evaluation\_metrics.py for detailed stats (mean, std\_dev, min/max).







Example Code Snippet

from mmseg.apis import init\_segmentor, inference\_segmentor

from utils.evaluation\_metrics import compute\_miou



model = init\_segmentor(config\_file, checkpoint\_file, device='cuda:0')

results = \[]

ground\_truths = \[]



\# Load test images and masks

for img\_path, mask\_path in test\_pairs:

&nbsp;   result = inference\_segmentor(model, img\_path)

&nbsp;   results.append(result)

&nbsp;   ground\_truths.append(load\_mask(mask\_path))



miou = compute\_miou(results, ground\_truths)

print(f"mIoU: {miou:.2%}")



Interpreting Results



Compare against baselines in Statistical\_Analysis\_Complete.xlsx.

Analyze generalization gap: Source mIoU vs. Target mIoU.

Visualize results using utils/visualization\_utils.py.



For reproduction, see reproduction\_guide.md.

