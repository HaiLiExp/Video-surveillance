# Video-surveillance

The goal of this project is to realize a remote video surveillance tool. 

Images are taken by a smartphone with an APP named "IP Webcam". The images are displayed in a browser on a remote computer but in the same WIFI network, i.e. the same SSID. The images in the browser are read and processed on this remote computer. The analysis results are sent to browser agains so that another remote computer can check them simultaneously. The image process task is a typical classification problem for machine learning. A black object is placed in a white background. The placement is correct if the black object is completely surrounded by white background. If the black object is placed beyond the border of the white background, the placement is wrong.

Hardware requirement:
A smartphone
A PC
A WIFI environment

Software requirement:
"IP Webcam" installed on the smartphone
Python, OpenCV, Tensorflow, Flask, browser

Structure of the code "IPCamPredSteaming.py":
Image acquisition:
The images taken by the video system are streamed by Android APP « IP camera » from smartphone to an URL, and further accessed by a web browser on a PC/cluster. 
The images are read by the python « requests »  module into the code for further pre-processing

Image pre-processing:
The images are read in the binary format and in the 2D gray scale image. 
The images are resized to a much smaller size to save the computation cost.
The images are further converted to the image format required by the Tensorflow.

Classification:
The pre-trained machine learning model is loaded, and it is used for the binary classification.
The result « OK » or « NOK », as well as the date and time are displayed on the image.
To improve the speed of computation, multiple threads are created to maximize the utilization of CPU

Broadcasting:
The result image is broadcasted to a IP address by flask and reviewed by a web browser.
The result can be reviewed simultaneously by multiple PCs. 

The screen output of the code ist exemplarily shown in Issue "OK" and "NOK" for the two classification results. The classification result "OK" or "NOK", the date and time is shown on the image, the commandline shows the output values of the machine learning model: less than 0.5 is "NOK", greater than 0.5 is "OK".

The code "trainModel.py" is to train the machine learning model on colab. The architecture of the model is found by trial and error, and its size is kept as small as possible and as large as necessary. The training accuracy and training loss are ploted. The last part of the code is for prediction with the training data for verification of the correctness of the execution.
