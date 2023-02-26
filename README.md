Summary (must keep to show summary outside of paper):
Detecting synthetic media has been an ongoing concern over the recent years due to the increasing amount of deepfakes on the internet. In this project, we will explore the different methods and algorithms that are used in deepfake detection.

Introduction: 
Deepfakes, or artificial intelligence-generated videos that depict real people doing and saying things they never did, have become a growing concern in recent years. These artificially generated content can be used to spread misinformation, manipulate public opinion, and even harm individuals. Therefore, the ability to detect deepfakes is crucial to ensure the integrity of information and protect people from potential harm.

Proposal:
The main objective of this project is to develop and evaluate advanced machine learning techniques for deepfake detection. Specifically, the project aims to investigate and analyze the current state-of-the-art deepfake detection methods, and evaluate the performance of the developed models using a dataset of deepfake videos. 

Datasets:
Deepfake Detection Challenge (DFDC) 
The DFDC (Deepfake Detection Challenge) is a Facebook developed dataset for deepface detection consisting of more than 100,000 videos. It is currently the largest publicly available dataset and was created for a competition aimed towards creating new and better models to detect manipulated media. The dataset consists of a preview dataset with 5k videos featuring two facial modification algorithms and a full dataset with 124k videos featuring 8 facial modification algorithms. 
https://ai.facebook.com/datasets/dfdc/

Celeb-DF
Celeb-DF is a dataset used for deepfake forensics. It includes 590 original videos collected from YouTube with subjects of different ages, ethnic groups and genders, and 5639 correspondingDeepFake videos. Unlike most other DeepFake datasets, Celeb-DF contains high visual quality videos that better resemble DeepFake videos circulated on the Internet. 
https://arxiv.org/abs/1909.12962
https://github.com/yuezunli/celeb-deepfakeforensics

Potential Architectures (Name 4 models, and a one sentence):
EfficientNet B1 LTSM
This is an implementation Efficient Net that was implemented for the DeepFake Detection Model
“To make it comparable with ResNet50 + LSTM it uses the same fully connected layers and also uses 512 hidden units as it was described in the paper”
DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection (https://arxiv.org/abs/2001.03024)
https://github.com/CatoGit/Comparing-the-Performance-of-Deepfake-Detection-Methods-on-Benchmark-Datasets/blob/master/deepfake_detector/pretrained_mods/efficientnetb1lstm.py
MesoNet 
The MesoInception4 deepfake detection architecture as introduced in MesoNet: a Compact Facial Video Forgery Detection Network (https://arxiv.org/abs/1809.00888) from Darius Afchar, Vincent Nozick, Junichi Yamagishi, Isao Echizen
ResNet LTSM
Implementation of a Resnet50 + LSTM with 512 hidden units as it was described in the paper
DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection (https://arxiv.org/abs/2001.03024)
parts from https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/2
and from https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/6 
XCeption
Creates an Xception Model as defined in:
Francois Chollet, Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

// To be completed after finishing implementation 
ResNet LSTM Implementation:
data augmentation
hyperparameters
“code”



Results:

Conclusion:

Demo:
https://github.com/jchangz01/CS188-Project-Deepfake-Detection



References:
