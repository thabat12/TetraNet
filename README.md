# Project TetraNet Overview

![image](https://user-images.githubusercontent.com/65915193/112793462-f5b15d80-902a-11eb-874b-decbbb800f6d.png)

Project TetraNet is a novel research project used for wildfire mitigation using footage from rapid nanosatellite deployment into space. We created a nanosatellite that utilizes computer vision models, image segmentation U-Net convolutional networks, linear regression artifical neural networks, and mathematical fire spread simulators in order to accurately apply wildfire patterns to a real-world setting, allowing researchers to prevent wildfires from spreading in high-risk regions.

# Nanosatellite Construction

![image](https://user-images.githubusercontent.com/65915193/113235068-11557780-9268-11eb-88a5-1da251bd10b5.png)
![image](https://user-images.githubusercontent.com/65915193/113235206-482b8d80-9268-11eb-9f64-8d097736f1ef.png)

We engineered a nanosatellite to deploy into the Earth's atmosphere from scratch. More information can be found [here](https://drive.google.com/file/d/19KJ8iIdx7iHpYULTgR8Wu4Gu_hYKOD-A/view?usp=sharing).


# Image Segmentation Analysis

![image](https://user-images.githubusercontent.com/65915193/113232996-d9e4cc00-9263-11eb-9a9b-24c5d19d305e.png)

The above image was annotated using Apeer to produce the image below. All of our images were obtained through our original footage from our launch to space in 2021 and then annotated similarly using Apeer. 

![image](https://user-images.githubusercontent.com/65915193/113232418-b40af780-9262-11eb-926d-491324db32b6.png)

Afterward, we trained the image segmentation U-Net convolution networks using similar annotated images. After applying our trained U-Net, it produced the image below.


# Predicting Burned Area using forest fire data set

Using the [forest fires data set](http://archive.ics.uci.edu/ml/datasets/Forest+Fires) from the UC Irvine Machine Learning Repository, we created a linear regression artificial neural network trained on this [ForestFires.csv](http://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv), which illustrates the hectares of area burned in a forest fire using input values like temperature, relative humidity, wind, and rain as well as the FFMC, DMC, DC, and ISI indexes.


## Installation

```
pip install 
```

## Usage

## Contributing

