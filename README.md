# sign2text
### Real-time AI-powered translation of American sign language to text

The project focuses on translating American Sign Language (ASL) [fingerspelled alphabet](http://lifeprint.com/asl101/topics/wallpaper1.htm) (26 letters). I utilised transfer learning to extract features, followed by a custom classification block to classify letters. This model is then implemented in a real-time system with OpenCV - reading frames from a web camera and classifying them frame-by-frame. This repository contains the code & weights for classifying the American Sign Language (ASL) alphabet in real-time.

This project was developed as my portfolio project at the Data Science Retreat (Batch 09) in Berlin. Please feel free to fork/comment/collaborate! Presentation slides are available in the repo :)

# Usage 

The entire pipeline (web camera -> image crop -> pre-processing -> classification) can be executed by running the live_demo.py script.

The live_demo.py script loads a pre-trained model ([VGG16](https://keras.io/applications/#vgg16)/[ResNet50](https://keras.io/applications/#resnet50)/[MobileNet](https://keras.io/applications/#mobilenet)) with a custom classification block, and classifies the ASL alphabet frame-by-frame in real-time. The script will automatically access your web camera and open up a window with the live camera feed. A rectangular region of interest (ROI) is shown on the camera feed. This ROI is cropped and passed to the classifier, which returns the top 3 predictions. The largest letter shown is the top prediction, and the bottom 2 letters are the second (left) and third (right) most probable predictions. The architecture of the classification block will be described further in Sections 4/5.

## Dependencies
The code was developed with python 3.5 and has been tested with the following libraries/versions:

- OpenCV 3.1.0
- Keras 2.0.8
- tensorflow 1.11 (cpu version), it will also run with the gpu-version
- numpy 1.15.2
- joblib 0.10.3

NOTE - feature extraction using the pre-trained models in Keras was run on an AWS EC2 p2.8xlarge instance with the [Bitfusion Ubuntu 14 TensorFlow-2017 AMI](https://aws.amazon.com/marketplace/pp/B01EYKBEQ0). Packages had to be manually updated, and Python 2 is the standard version. You can either update to Python 3, or edit the scripts to work with Python 2 (the only issues should be the print statements)

## Running the Live Demo
   
When running the script, you must choose the pre-trained  model you wish to use. You may optionally load your own weights for the classification block. 

```bash
$ python live_demo.py --help
usage: live_demo.py [-h] [-w WEIGHTS] -m MODEL

optional arguments:
  -h, --help            show this help message and exit
  -w WEIGHTS, --weights WEIGHTS
                        path to the model weights

required arguments:
  -m MODEL, --model MODEL
                        name of pre-trained network to use
```

NOTE - On a MacBook Pro (macOS SIERRA 16GB 1600MHz DDR3/2.2 GHz Intel Core i7) using the CPU only, it can take up to ~250ms to classify a single frame. This results in lag during real-time classification as the effective frame rate is anywhere from 1-10 frames per second,  depending on which model is running. MobileNet is the most efficient model. Performance for all models is is significantly improved if running on a GPU. 

# 1. American Sign Language

There are no accurate measurements of how many people use American Sign Lanuage (ASL) - estimates vary from 500,000 to 15 million people. However, 28 million Americans (~10% of the population) have some degree of hearing loss, and 2 million of these 28 million are classified as deaf. For many of these people, their first lanugage is ASL.

The ASL alphabet is 'fingerspelled' - this means all of the alphabet (26 letters, from A-Z) can be spelled using one hand. There are 3 main use cases of fingerspelling in any sign language: 

(i) Spelling your name
(ii) Emphasising a point (i.e. literally spelling out a word)
(iii) When saying a word not present in the ASL dictionary (the current Oxford English dictionary has ~170,000 words while estimates for ASL range from 10,000-50,000 words)

This project is a (very small!) first step towards bridging the gap between 'signers' and 'non-signers'.

# 2. Pre-processing
coming soon I promise
# 3. Transfer learning & feature extraction
coming soon
# 4. Training
coming soon
# 5. Real-time system
coming soon

# 6. References
https://research.gallaudet.edu/Publications/ASL_Users.pdf
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

