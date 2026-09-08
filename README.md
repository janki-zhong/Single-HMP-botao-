## 运动学引导的双流模糊学习的人体动作预测
This is the code for the paper

Botao Zhou, Jianqi Zhong, Wenming Cao, Wenbin Zou. 
[_Kinematics-guided Dual-stream Fuzzy Learning for Human Motion Prediction_]. In 计算机应用.

### Dependencies

* cuda 12.1
* Python 3.10
* [Pytorch] 2.1.0

### Get the data

[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).

Directory structure: 
```shell script
H3.6m
|-- S1
|-- S5
|-- S6
|-- ...
`-- S11
```
[AMASS](https://amass.is.tue.mpg.de/en) from their official website..

Directory structure:
```shell script
amass
|-- ACCAD
|-- BioMotionLab_NTroje
|-- CMU
|-- ...
`-- Transitions_mocap
```
[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) from their official website.

Directory structure: 
```shell script
3dpw
|-- imageFiles
|   |-- courtyard_arguing_00
|   |-- courtyard_backpack_00
|   |-- ...
`-- sequenceFiles
    |-- test
    |-- train
    `-- validation
```
Put the all downloaded datasets in ./datasets directory.

### Training
All the running args are defined in [opt.py](utils/opt.py). We use following commands to train on different datasets and representations.
To train,
```bash
python main_h36m_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 10 --skip_rate 1 --batch_size 128 --test_batch_size 128 --in_features 66 --num_stage 12
```
```bash
python main_h36m_ang.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 10 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 48
```
```bash
python main_amass_3d.py --kernel_size 10 --dct_n 35 --input_n 50 --output_n 25 --skip_rate 5 --batch_size 128 --test_batch_size 128 --in_features 54 
```
### Evaluation
To evaluate the pretrained model,
```bash
python main_h36m_3d_eval.py --is_eval --kernel_size 10 --dct_n 20 --input_n 50 --output_n 10 --skip_rate 1 --batch_size 64 --test_batch_size 128 --in_features 66 --ckpt checkpoint/main_h36m_3d_in50_out10_ks10_dctn20
python main_h36m_3d_eval.py --is_eval --kernel_size 10 --dct_n 20 --input_n 50 --output_n 25 --skip_rate 1 --batch_size 64 --test_batch_size 128 --in_features 66 --ckpt checkpoint/main_h36m_3d_in50_out25_ks10_dctn20
```
```bash
python main_h36m_ang_eval.py --is_eval --kernel_size 10 --dct_n 20 --input_n 50 --output_n 25 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 48 --ckpt ./checkpoint/pretrained/h36m_ang_in50_out10_dctn20/
```
```bash
python main_amass_3d_eval.py --is_eval --kernel_size 10 --dct_n 35 --input_n 50 --output_n 25 --skip_rate 5 --batch_size 128 --test_batch_size 128 --in_features 54 --ckpt ./checkpoint/pretrained/amass_3d_in50_out25_dctn30/
```

### Citing

If you use our code, please cite our work


### Acknowledgments
The overall code framework (dataloading, training, testing etc.) is adapted from [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline). 

The predictor model code is adapted from [LTD](https://github.com/wei-mao-2019/LearnTrajDep).

Some of our evaluation code and data process code was adapted/ported from [Residual Sup. RNN](https://github.com/una-dinosauria/human-motion-prediction) by [Julieta](https://github.com/una-dinosauria). 

### Licence
MIT
