# sign2text
### Real-time AI-powered translation of American sign language to text

This repository contains the code & weights for classifying American Sign Language (ASL) alphabet in real-time using deep learning. The project focuses on American Sign Language (ASL), and specifically the ASL alphabet (26 letters). I utilised transfer learning to extract features followed by a custom classification block. 

This project was developed as my portfolio project at the Data Science Retreat (Batch 09) in Berlin. Please feel free to fork/comment/collaborate!

# Usage 

The entire pipeline (web camera -> image crop -> pre-processing -> classification) can be executed by running the live_demo.py script.

The live_demo.py script loads a pre-trained model (VGG16 or ResNet50) with a custom classification block, and classifies American Sign Language fingerspelling frame-by-frame in real-time. The script will automatically access your web camera and open up a window with the live camera feed. A rectangular region of interest (ROI) is shown on the camera feed; this ROI is cropped and passed to the classifier, which returns the top 3 letter predictions. The largest letter shown is the top prediction, and the bottom 2 letters are the second (left) and third (right) most probable predictions. The architecture of the classification block is described further in Sections 4/5.

## Dependencies
The code was developed with python 3.5 and requires the following libraries/versions:

- OpenCV 3.0.0
- keras 2.0.1
- tensorflow-gpu 1.0.1 (It can also work with non gpu version)
- numpy 1.12.0
- joblib 0.10.3

## Running the Live Demo
   
When running the script, you must choose the pre-trained  model you wish to use. You may optionally load your own weights for the classification block. 

```bash
$ python live_demo.py --help
usage: live_demo.py [-h] -m MODEL [-w WEIGHTS]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model     MODEL
                        name of pre-trained network to use
  -w WEIGHTS, --weights WEIGHTS
                        path to the model weights

```

NOTE - On a MacBook Pro (macOS SIERRA 16GB 1600MHz DDR3/2.2 GHz Intel Core i7), it takes ~250ms/frame to classify resulting in  lag during real-time classification. The effective frame rate is only 5-10 frames can effectively be processed; however, this is significantly improved if running on a GPU.

# 1. American Sign Language

There are no accurate measurements of how many people use American Sign Lanuage (ASL) - estimates vary from 500,000 to 15 million people. However, 28 million Americans (~10% of the population) have some degree of hearing loss, and 2 million of these 28 million are classified as deaf. For many of these people, their first lanugage is ASL.

The ASL alphabet is 'fingerspelled' - this means all of the alphabet (26 letters, from A-Z) can be spelled using one hand. There are 3 main use cases of fingerspelling in any sign language: 

(i) Spelling your name
(ii) Emphasising a point (i.e. literally spelling out a word)
(iii) When saying a word not present in the ASL dictionary (the current Oxford English dictionary has ~170,000 words while estimates for ASL range from 10,000-50,000 words)

# 2. Pre-processing
coming soon
# 3. Transfer learning & feature extraction
coming soon
# 4. Training
coming soon

# 5. References
https://research.gallaudet.edu/Publications/ASL_Users.pdf
