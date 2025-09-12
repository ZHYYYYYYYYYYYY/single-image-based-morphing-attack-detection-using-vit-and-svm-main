# Single Image Based Face Morphing Attack Detection Using Vision Transformer and SVM

# Requirement
For details, please check [requirements.txt](./requirements.txt)

## Instruction
Tested on:
    Ubuntu 20.04 LTS
    Python 3.8.10
    Nvidia A100 GPU

0. Install TensorFlow 1 for MTCNN, you can also use other pytorch implementations.
    ```
    pip install nvidia-pyindex==1.0.9
    pip install nvidia-tensorflow[horovod]==1.15.5+nv21.3
    pip install protobuf==3.20.1
    pip install mtcnn==0.1.1
    ```

1. Install Pytorch 1.8.2 LTS fits your cuda version from: [pytorch.org](https://pytorch.org/get-started/previous-versions/)
    ```
    pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
    ```
2. Install other dependencies
    ```
    pip install tqdm==4.66.1
    pip install scikit-learn==0.23.0
    pip install pandas==1.0.0
    pip install numpy==1.19.5
    pip install opencv-python==4.7.0.72
    ```
# How to Run

## Crop Images Using MTCNN
--source_dir: a path of folder of non-cropped images

--target_dir: a path of folder to store MTCNN-cropped images

--source_suffix: suffix of non-cropped images

    ```
    ./src/crop_by_MTCNN.py --source_dir PATH_INPUT_FOLDER --target_dir PATH_OUTPUT_FOLDER --source_suffix png
    ```

## Extract ViT features

Download the pre-trained ViT model from: [GoogleDrive](https://drive.google.com/file/d/17_2NMniZ7OoLsCWaHCRxpZ3DKtROFyi2/view?usp=sharing), and by default place it to [./experiments/Pretrained_Imagenet_MTCNN/save/imagenet21k+imagenet2012_ViT-L_32.pth](./experiments/Pretrained_Imagenet_MTCNN/save/).

Extract embeddings given a path of image folder.

--data-dir: a path of folder of input images

--output-dir: a path of folder to store extracted embeddings and filename records

Other arguments are for the ViT model.

    ```
    python ./src/eval_extract_embeddings_1inputdir.py --n-gpu 1 --model-arch l32 --checkpoint-path experiments/Pretrained_Imagenet_MTCNN/save/imagenet21k+imagenet2012_ViT-L_32.pth --image-size 384 --batch-size 32 --num-workers 8 --data-dir PATH_INPUT_FOLDER --output-dir PATH_OUTPUT_FOLDER --num-classes 1000 --part test
    ```
## SVM for MAD
Check the [train_LinearSVM_MAD.py](./src/train_LinearSVM_MAD.py) for an example script to train SVM and store the trained model.

Check the [eval_LinearSVM_MAD.py](./src/eval_LinearSVM_MAD.py) for an example script to load a trained SVM model and run evaluation.
This script will generate a txt file containing filename and scores, which can be used for further evluation to calculate MACER/BPCER, or plot the DET curve.


# Reference

```
@InProceedings{Zhang_2024_CVPR,
    author    = {Zhang, Haoyu and Ramachandra, Raghavendra and Raja, Kiran and Busch, Christoph},
    title     = {Generalized Single-Image-Based Morphing Attack Detection Using Deep Representations from Vision Transformer},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {1510-1518}
}
```


