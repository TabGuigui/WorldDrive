# Installation for WorldDrive

After successfully downloading the NAVSIM dataset:

## 1. Important Version
cuda==12.1 \
torch==2.3.0 \
deepspeed==0.14.4 \
transformers==4.56.2 \
diffusers==0.34.0

## 2. TA-DWM pretrain
Download the pretrained world model checkpoint [Model](https://huggingface.co/tabguigui/WorldDrive/tree/main)

## 3. Planner training & eval

python3 setup.py develop

And see [WorldDrive Training and Evaluation](Train_Eval.md)