# Source Code of MEGA Model for Multimodal Relation Extraction
Implementation of Our Paper "[Multimodal Relation Extraction with Efficient Graph Alignment](https://dl.acm.org/doi/abs/10.1145/3474085.3476968)" in ACM Multimedia 2021. This Implementation is based on the [OpenNRE](https://github.com/thunlp/OpenNRE).

## Model Architecture
![model](model.png)

The Overall Framework of Our Proposed MEGA Model. Our Model Introduces Visual Information into Predicting Textual Relations. Besides, We leverages the Graph Structural Alignment and Semantic Alignment to Help Model Find the Mapping From Visual Relations to Textual Contents.


## Requirements
* `torch==1.6.0`
* `transformers==3.4.0`
* `pytest==5.3.2`
* `scikit-learn==0.22.1`
* `scipy==1.4.1`
* `nltk==3.4.5`

## Data Format
The MNRE dataset used in our paper can be downloaded [here](https://drive.google.com/file/d/1gD9ipQgDEDRxaVxkKr8T0gFFQgKyPpa7/view?usp=sharing). Unzip and move it to `./benchmark/ours/`.

For more information regarding the dataset, please refer to the [MNRE](https://github.com/thecharm/MNRE) repository. 

>Each sentence is split into several instances (depending on the number of relations).
>Each line contains
>```
>'token': Texts preprocessed by a tokenizer
>'h': Head entities and their positions in a sentence
>'t': Tail entities and their positions in a sentence
>'image_id': You can find the corresponding images using the link above
>'relation': The relations and entity categories
>```



## Usage
### Training
You can train your own model with OpenNRE. In `example` folder we give the training codes named by `train.py` for MEGA. You can use the following  script to train a MEGA model on the MNRE dataset.
>```
>python example/train.py \
>--dataset ours \
>--max_epoch 10 \
>--batch_size 32 \
>--metric micro_f1 \
>--lr 2e-5 \
>--ckpt MEGA
>```

Note that a pretrained BERT weights should be used for initialization, which you can download [here](https://drive.google.com/file/d/1HYWznU1rjNiHr1aoNq7vOWfnzatdTnFL/view?usp=sharing) and put it in `./`.

### Inference
Besides, we provide the pretrained checkpoint for quick inference which you can download from [here](https://drive.google.com/file/d/1HYWznU1rjNiHr1aoNq7vOWfnzatdTnFL/view?usp=sharing)
To run MEGA model in inference mode, you can add the `--only_test` parameter to the script above and edit the `--ckpt` parameter by the name of provided pretrained checkpoint. By the way, you should move the pretrained checkpoints to the `ckpt` folder for inference.
>```
>python example/train.py \
>--dataset ours \
>--batch_size 32 \
>--metric micro_f1 \
>--only_test \
>--ckpt pretrained_MEGA
>```

## Citation
If you find this repo helpful, please cite the following:
```latex
@inproceedings{zheng2021multimodal,
  title={Multimodal Relation Extraction with Efficient Graph Alignment},
  author={Zheng, Changmeng and Feng, Junhao and Fu, Ze and Cai, Yi and Li, Qing and Wang, Tao},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={5298--5306},
  year={2021}
}

@inproceedings{zheng2021mnre,
  title={MNRE: A Challenge Multimodal Dataset for Neural Relation Extraction with Visual Evidence in Social Media Posts},
  author={Zheng, Changmeng and Wu, Zhiwei and Feng, Junhao and Fu, Ze and Cai, Yi},
  booktitle={2021 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}
```
