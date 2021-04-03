# Project TetraNet Overview

![image](https://user-images.githubusercontent.com/65915193/112793462-f5b15d80-902a-11eb-874b-decbbb800f6d.png)

Project TetraNet is a novel research project used for wildfire mitigation using footage from rapid nanosatellite deployed into space. We created a nanosatellite that utilizes computer vision models, image segmentation U-Net convolutional networks, linear regression artifical neural networks, and mathematical fire spread simulators in order to accurately apply wildfire patterns to a real-world setting, allowing researchers to prevent wildfires from spreading in high-risk regions.

## Nanosatellite Engineering

![image](https://user-images.githubusercontent.com/65915193/113235068-11557780-9268-11eb-88a5-1da251bd10b5.png)
![image](https://user-images.githubusercontent.com/65915193/113235206-482b8d80-9268-11eb-9f64-8d097736f1ef.png)

![image](https://user-images.githubusercontent.com/65915193/113466894-e5bbc400-9404-11eb-81d6-a066f42b0cdb.png)

We engineered a nanosatellite to deploy into the Earth's atmosphere from scratch. More information can be found [here](https://drive.google.com/file/d/19KJ8iIdx7iHpYULTgR8Wu4Gu_hYKOD-A/view?usp=sharing).

## Image Segmentation Analysis using U-Net Convolutional Neural Network

Our TetraNet nanosatellite conducts advanced image segmentation analysis to determine the terrain and topology of the land through scanning for dense vegetation on the ground. The images below provide insight into how we created the image segmentation U-Net convolutional neural network.

![image](https://user-images.githubusercontent.com/65915193/113466073-83f85b80-93fe-11eb-8fd1-9d46f0f20d39.png)

The above image was annotated using Apeer to produce the image below. All of our images were obtained through our original footage from our launch to space in 2021 and then annotated similarly using Apeer. 

![image](https://user-images.githubusercontent.com/65915193/113466742-b48ec400-9403-11eb-90d6-e0942d18d397.png)

Afterward, we converted our annotated Apeer image to the "mask" that we could use to train the image segmentation U-Net convolutional network.

![image](https://user-images.githubusercontent.com/65915193/113466307-62986f00-9400-11eb-9ea6-c8c17a10446e.png)

After applying our trained U-Net to the above image, it produced the "predicted mask" below, showing how our image segmentation model was highly accurate in detecting dense vegetation. 

![image](https://user-images.githubusercontent.com/65915193/113466312-6b894080-9400-11eb-905d-58190ceb45be.png)

Lastly, we overlayed the "predicted mask" images on the original images to further prove how accurate the model was.

![image](https://user-images.githubusercontent.com/65915193/113466824-4696cc80-9404-11eb-92e9-0528f5dac718.png)

## Predicting Burned Area using Artifical Neural Network

Using the [forest fires data set](http://archive.ics.uci.edu/ml/datasets/Forest+Fires) from the UC Irvine Machine Learning Repository, we created a linear regression artificial neural network trained on this [ForestFires.csv](http://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv), which illustrates the hectares of area burned in a forest fire based on input values like temperature, relative humidity, wind, and rain as well as the FFMC, DMC, DC, and ISI indexes.

![image](https://user-images.githubusercontent.com/65915193/113236263-29c69180-926a-11eb-9d14-76c16691f2c6.png)

## Architecture Diagram

![image](https://user-images.githubusercontent.com/65915193/113466753-ce300b80-9403-11eb-95f3-774be12129f1.png)

![image](https://user-images.githubusercontent.com/65915193/113466758-d4be8300-9403-11eb-80f4-87d64cfdbc1c.png)

## Fire Spread Simulation

![image](https://assets.website-files.com/5f45dcafd2144b042ed84cfd/5f45fb0ddb2e8ed9d9f3b976_7e44396cc5f3f0e8d3fdceb51be326fc4b25110e.gif)

We utilized a mathematical model to depict how fire spreading works while wind is blowing. The results of the code can be seen here.  

## Web Application

We utilized the Python Flask back-end framework to make API calls to Azure's machine learning and cloud services, providing an efficient method of conducting analysis on the footage.

## Utilizing Azure Machine Learning and Cloud Services 

## Practical Applications & Future Implications

Our project is a low-cost alternative to traditional satellites deployed into space that track geographical features and trends. Our satellite can return footage back to Earth to conduct computer vision analysis on certain terrain to analyze its susceptability to wildfires, especially during the threatening summer season. With the trending temperature increase as a result of global warming, our nanosatellite can provide substantial impact by possibly providing real-time alerts to fire departments and public safety services across the world to not only mitigate wildfires and save lives but also prevent severe smoke inhalation and save millions of dollars in damage caused by wildfires annually.

## Contributors

Project TetraNet was created by Abhinav Bichal, Sarvesh Sathish, and Pranay Shah for the 2021 Microsoft Azure AI competition.
