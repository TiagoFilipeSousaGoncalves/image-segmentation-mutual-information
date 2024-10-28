# Deep Image Segmentation based on Mutual Information: A Study

## About
Implementation of the paper [_"Deep Image Segmentation based on Mutual Information: A Study"_](paper.pdf) by Tiago Gonçalves, Leonardo Capozzi, Ana Rebelo and Jaime S. Cardoso.



## Abstract 
The use of deep neural networks has achieved many advancements in many different areas, namely computer vision. Information theory concepts such as the Kullback-Leibler Divergence is often used in deep learning methodologies as optimisation criterion since it quantifies the difference between two probability distributions. Image segmentation is a computer vision problem where the goal is to classify individual pixels of an image. It has many real-world applications such as self-driving cars, medical imaging and object detection, to name a few. In this paper, we present a comparative study of two methodologies based on mutual information applied to the task of deep image segmentation.



## Clone this repository
To clone this repository, open a Terminal window and type:
```bash
$ git clone git@github.com:TiagoFilipeSousaGoncalves/image-segmentation-mutual-information.git
```
Then go to the repository's main directory:
```bash
$ cd image-segmentation-mutual-information
```



## Dependencies
### Install the necessary Python packages
We advise you to create a virtual Python environment first (Python 3.7). To install the necessary Python packages run:
```bash
$ pip install -r requirements.txt
```



## Data
To know more about the data used in this paper, please send an e-mail to  [**tiago.f.goncalves@inesctec.pt**](mailto:tiago.f.goncalves@inesctec.pt) or to [**leonardo.g.capozzi@inesctec.pt**](mailto:leonardo.g.capozzi@inesctec.pt).



## Usage
### Train Models
To reproduce the experiments:
```bash
$ python code/train_unet.py
$ python code/train_unet_rmi.py
$ python code/train_unet_deeplabv3_mutual.py
```

### Test Models
To reproduce the experiments:
```bash
$ python code/test_unet.py
$ python code/test_unet_rmi.py
$ python code/test_unet_deeplabv3_mutual.py
```

### Generate the Results
To plot the results:
```bash
$ python code/generate_imgs_unet.py
$ python code/generate_imgs_unet_rmi.py
$ python code/generate_imgs_unet_deeplabv3_mutual.py
```



## Citation
If you use this repository in your research work, please cite this paper:
```bibtex
@inproceedings{goncalvescapozzi2024recpad,
	author = {Tiago Gonçalves, Leonardo Capozzi, Ana Rebelo and Jaime S. Cardoso},
	title = {{Deep Image Segmentation based on Mutual Information: A Study}},
	booktitle = {{30th Portuguese Conference in Pattern Recognition (RECPAD)}},
	year = {2024},
	address = {{Covilhã, Portugal}}
}
```