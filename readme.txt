Suppose the environment is windows and run command through Visual Studio Code's (VSC) terminal.
Assume python 3.9 is already installed and chosen as default python interpretor for VSC
Assume you are running with a PC with cuda installed. (R.I.P AMD)
Note that dataset for part 2, model used for part 3 and 4 can be downloaded at 
https://drive.google.com/drive/u/0/folders/1cdZXZsz8Acl_kaS8P9zEUQv41g2oHPgG

1. Install the environment
1.1 pytorch:       $ python3.9.exe -m pip install torch
1.2 torchvision:   $ python3.9.exe -m pip install torchvision

2. Train CIFAR10 and CIFAR100 
2.1 go to directory of ECE740:  $ cd PATH_TO_FOLDER
2.2 CIFAR10: python3.9.exe:     $ python3.9.exe train_cifar10.py
2.3 CIFAR100: python3.9.exe:    $ python3.9.exe train_cifar100.py
2.4 Result for part 1 can be verified here. Generated filename example: "./resnet18cifa10/model-widers-epoch186.pt"
Note that if you cannot download CIFAR10/CIFAR100 dataset through code because of 
urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
you can download data manually and put data into "./data" directory
example of filename: cifar-10-python.tar.gz

3. Use Autoattack to attack on CIFAR10 and CIFAR100
3.1 Install autoattack: $python3.9.exe -m pip install git+https://github.com/fra31/auto-attack
3.2 Change the 'path' variable to your resnet18 model trained with cifar10 in part 2.
Note that you MUST have CUDA available on the device if you are using the model downloaded from the link from googledrive provided.
model filename example: path = "./cifar10model.pt". 
If you want to replace to your own generated model in 2.4, this variable should be changed to path = "./resnet18cifa10/model-widers-epoch186.pt"
3.3 You can change the variable "n_ex" to a smaller value if you want to perform autoattack quicker. 
This variable will influence the size of attacked images. However, if you are using your own generated model,
for the performance for next part, we do not suggest to change the variable "n_ex" to a smaller value. 
Otherwise there will be a Zero Division Error caused by not enough training dataset.
3.4 If you want to change attack parameter to L2, change parameter "norm" to "L2"
3.5 Attack CIFAR10 with $ python3.9.exe attack_cifar10.py
3.6 Attack CIFAR100 following the procedures above
3.7 Result for part 2 can be verified here. Generated filename exammple: "./attackDir/aa_standard_50000_Linf_eps_0.03100.pth"

4. Use DeepSAD, an anomaly detection algorithm to investigate on AA attack detection
4.1 Go to DeepSAD directory by $ cd src
4.2 In the file "deep_10.py" line 93, replace the 'adversial_data_path' in 'AutoAttck_CIFAR10_Dataset'
 to the autoattacked filename generated from step 3.7. If you want to use the "aa_standard_50000_Linf_eps_0.03100.pth" file uploaded. 
 You should put it into directory "attackDir".
4.3 In the file "deep_10.py" line 108, replace the model path in deepSAD.load_Resnet_model(model_path=r"../cifar100-ckpt.pth") 
to the actual CHECKPOINT path generated in step 2.4.
4.4 Run DeepSAD for CIFAR10 with  $ python3.9.exe deep_10.py
4.5 Run DeepSAD for CIFAR100 following the procedures above

5. Overall directory structure for all parts (Modified files and input/output files only)
ECE740/
    ├── data/
    │   ├── cifar-10-python.tar.gz (original dataset for step 2.2)
    │   ├── cifar-100-python.tar.gz (original dataset for step 2.3)
    ├── models/
    ├── src/
        ├── datasets/
            ├── cifar10.py
        ├── all-other-directories-from-DeepSAD-original-project/
        ├── deep_10.py
        ├── deep_100.py
    ├── resnet18cifa10/
        ├── model-wideres-epochxxx.pt (result in step 2.4 for CIFAR10; input for step 3.2)
    ├── resnet18cifa100/
        ├── model-wideres-epochxxx.pt (result in step 2.4 for CIFAR100)
    ├── checkpoint/
        ├── ckpt.pth (result in step 2.4 for CIFAR10; input for step 4.3)
    ├── checkpointcifa100/
        ├── ckpt.pth (result in step 2.4 for CIFAR100)
    ├── attack_cifar10.py
    ├── attack_cifar100.py
    ├── train_cifar10.py
    ├── train_cifar100.py
    ├── utils.py
    ├── readme.txt