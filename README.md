# FibUnet: Fibonacci-infused UNet for Video Frame Prediction
A video prediction model employing Convolutional Neural Networks, Multi-layer Perceptrons coupled with a Fibonacci-infusing rule to address one-frame prediction problem.
# Get Started
1. Download data from [this folder](https://unimelbcloud-my.sharepoint.com/:f:/g/personal/thingoctramt_student_unimelb_edu_au/EuuPlkBtJnxJq_XxsoNf8rIBzNG7_X_xt1-Zt2i_YskaRg?e=cjQ0Zt). Datasets are put in repositories with the same name.
2. Install Python 3.9 and Tensorflow 2.15.1.
3. Install packages
```
python -m pip install -r requirements.txt
```
3. Train the model
```
python unet_mlp_mnist.py
```
4. Test the model on the Moving MNIST dataset
```
python unet_mlp_mnist_test.py
```
You can get the pre-trained models on three datasets from [here](https://unimelbcloud-my.sharepoint.com/:f:/g/personal/thingoctramt_student_unimelb_edu_au/EiZwb2o84NdNvzlxCaDulccB8i1Wc_qLiWueJrbMnX2efQ?e=Dpish3)
# Citation
@misc{tran2024fibunet,
      title={{FibUnet}: Fibonacci-infused UNet for Video Frame Prediction}, 
      author={Tram Tran},
      year={2024},
}