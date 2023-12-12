#Low Category Uncertainty and High Training Potential Instance Learning for Unsupervised Domain Adaptation
##Basic setting
###File path
>The file folder path under the  LUHP\examples\domain_adaptation\image_classification\
Include:
>>* Method file: LUHP.py
>>* Runing file: LUHP_DomainNet.py, LUHP_Home.py and LUHP_Office.py

>The datasets used in the repository should be downloaded under the Dataset folder with corresponding data and their data_lists.
---
### Requirements
	<ul>
	<li>pytorch 1.7.1</li>
	<li>numpy 1.21.2</li>
	<li>torchvision 0.8.2</li>
	<li>tqdm 4.62.3</li>
	<li>timm 0.4.12</li>
	<li>scikit-learn 1.0.2</li>
	</ul>
---
### Training
Using the `LUHP_Office.py` for training on Office31 dataset can be found below.
```
python LUHP_Office.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 40 --seed 1 -i 500 --log logs/luhp/Office31_D2A
python LUHP_Office.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 40 --seed 1 -i 500 --log logs/luhp/Office31_W2A
python LUHP_Office.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --seed 1 -i 500 --log logs/luhp/Office31_A2D
python LUHP_Office.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 1 -i 500 --log logs/luhp/Office31_A2W
python LUHP_Office.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --seed 1 -i 500 --log logs/luhp/Office31_D2W
python LUHP_Office.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --seed 1 -i 500 --log logs/luhp/Office31_W2D
```

Using the `LUHP_Home.py` file for training. Sample command to execute the training of the LUHP methods on OfficeHome dataset (with Art as source domain and Product as the target domain) can be found below. 
```
python LUHP_Home.py data/office-home -d OfficeHome -s A -t C -a resnet50 --epochs 20 -i 500 --seed 1 --log logs/luhp/OfficeHome_A2C
```

Using the `LUHP_DomainNet.py` file for training. Sample command to execute the training of the LUHP methods on DomainNet dataset (with real as source domain and clipart as the target domain) can be found below.
```
python LUHP_DomainNet.py data/DomainNet -d DomainNet -s R -t C -a resnet50 --epochs 20 -i 500 --seed 1 --log logs/luhp/DoaminNet_R2C
```
---
### Overview of the arguments
Generally, all scripts in the project take the following flags
- `DIR`: dataset path(data/Office31|OfficeHome|DomainNet|VisDA2017|)
- `-a`: Architecture of the backbone. (resnet50|resnet101)
- `-d`: Dataset (|Office31|OfficeHome|DomainNet|VisDA2017) 
- `-s`: Source Domain
- `-t`: Target Domain
- `--epochs`: Number of Epochs to be trained for.
- `--log`: path of the run log.
- `-i`: iterations per epoch
---
### Acknowledgement
Our implementation is based on the Transfer Learning Library.

