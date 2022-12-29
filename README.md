# MOL
The official code of "MOL: Towards Accurate Weakly Supervised Remote Sensing Object Detection via Multi-view nOisy Learning".
![Image text](https://github.com/GC-WSL/MOL/MOL.png)

## Get Started-Stage 1
### Installation
```Shell
conda create -n MOL-1 python=3.6
pip install torch==1.4.0 torchvision==0.5.0
cd MOL-1/lib
bash make_cuda.sh
```
### Data Preparation
Download the NWPU VHR-10.v2/DIOR dataset and put them into the `./data` directory. For example:
```Shell
  ./data/NWPU/                           
  ./data/NWPU/Annotations
  ./data/NWPU/JPEGImages
  ./data/NWPU/ImageSets    
```
Utilizing the selective search tools implemented in opencv-python to generate candidate proposals and put them into the `./data/selective_search_data`. For example:
```Shell
  ./selective_search_data/NWPU_train.mat                           
  ./selective_search_data/NWPU_val.mat
  ./selective_search_data/NWPU_test.mat  
```

### Training
```Shell
bash ./configs/NWPU/train_30000.sh 0 NWPU vgg16 model_name
```
### Testing
```Shell
bash ./configs/NWPU/test_30000.sh 0 NWPU vgg16 model_name
```
