# FibUnet: Fibonacci-infused UNet for Video Frame Prediction
A video prediction model employing Convolutional Neural Networks, Multi-layer Perceptrons coupled with a Fibonacci-infusing rule to address one-frame prediction problem.
# Get Started
1. Download data from [this folder](https://drive.google.com/drive/folders/1Z7r8lkSOeqVEPiMz4B6TaOeUVN6kqU_a?usp=drive_link). Datasets are put in repositories with the same name.
2. Install Python 3.9 and Tensorflow 2.15.1.
3. Install packages
```
python -m pip install -r requirements.txt
```
3. Train the model on the Moving MNIST dataset
```
python fibunet_mnist.py
```
4. Test the model on the Moving MNIST dataset
```
python fibunet_mnist_test.py
```
Prediction interval is controlled by the variable `OUTPUT_INDEX` at the top of training and testing files. Pre-trained models on three datasets can be downloaded from [here](https://drive.google.com/drive/folders/16AckbOYiHfBCrhsInYWX4nJEzKvCHqob?usp=sharing).
# Citation
```
@misc{tran2024fibunet,
      title={{FibUnet}: Fibonacci-infused UNet for Video Frame Prediction}, 
      author={Tram Tran},
      year={2024},
}
```
