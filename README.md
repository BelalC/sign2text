# sign2text
### Real-time AI-powered translation of American sign language to text

The project focuses on translating American Sign Language (ASL) [fingerspelled alphabet](http://lifeprint.com/asl101/topics/wallpaper1.htm) (26 letters). I utilised transfer learning to extract features, followed by a custom classification block to classify letters. This model is then implemented in a real-time system with OpenCV - reading frames from a web camera and classifying them frame-by-frame. This repository contains the code & weights for classifying the American Sign Language (ASL) alphabet in real-time.

This project was developed as my portfolio project at the Data Science Retreat (Batch 09) in Berlin. Please feel free to fork/comment/collaborate! Presentation slides are available in the repo :)

Dataset - https://drive.google.com/drive/folders/1-t8rgN3eOW99KGDy7U0HJhrbbwOe-5Wh?usp=sharing

All the data is already split into train/validation subsets, and labelled with letters from A-Z. 

NOTE - the Massey dataset I've included is already pre-processed and is only a subset of the entire dataset (part 5). I added padding due to odd shaped images, and also dropped a colour channel as there was a lot of green screen background in the images. Dropping the colour channel didn't cause any significant changes in performance so I've left it in. You can get the raw data directly from Massey University.

# Usage 

The entire pipeline (web camera -> image crop -> pre-processing -> classification) can be executed by running the live_demo.py script.

The live_demo.py script loads a pre-trained model ([VGG16](https://keras.io/applications/#vgg16)/[ResNet50](https://keras.io/applications/#resnet50)/[MobileNet](https://keras.io/applications/#mobilenet)) with a custom classification block, and classifies the ASL alphabet frame-by-frame in real-time. The script will automatically access your web camera and open up a window with the live camera feed. A rectangular region of interest (ROI) is shown on the camera feed. This ROI is cropped and passed to the classifier, which returns the top 3 predictions. The largest letter shown is the top prediction, and the bottom 2 letters are the second (left) and third (right) most probable predictions. The architecture of the classification block will be described further in Sections 4/5.

## Dependencies
The code + dependencies have been modernised recently (thanks Claude :D). It requires Python 3.10+ and the following libraries:

- tensorflow 2.14.0 (includes Keras)
- opencv-python 4.8.1
- numpy 1.24.3
- joblib 1.3.2

Install all dependencies with:

```bash
pip install -r requirements.txt
```

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

NOTE - On a MacBook Pro (macOS SIERRA 16GB 1600MHz DDR3/2.2 GHz Intel Core i7, yeah yeah I know it's 2026 & all) using the CPU only, it can take up to ~250ms to classify a single frame. This results in lag during real-time classification as the effective frame rate is anywhere from 1-10 frames per second,  depending on which model is running. MobileNet is the most efficient model. Performance for all models is significantly improved if running on a GPU. Feel free to play around with it.

# 1. American Sign Language

There are no accurate measurements of how many people use American Sign Lanuage (ASL), as the US census has never counted it - estimates vary from 500,000 to 15 million people. However, 28 million Americans (~10% of the population) have some degree of hearing loss, and 2 million of these 28 million are classified as deaf. For many of these people, their first lanugage is ASL.

The ASL alphabet is 'fingerspelled' - this means all of the alphabet (26 letters, from A-Z) can be spelled using one hand. There are 3 main use cases of fingerspelling in any sign language: 

(i) Spelling your name
(ii) Emphasising a point (i.e. literally spelling out a word)
(iii) When saying a word not present in the ASL dictionary (the current Oxford English dictionary has ~170,000 words while estimates for ASL range from 10,000-50,000 words)

This project is a (very small!) first step towards bridging the gap between 'signers' and 'non-signers'.

# 2. Pre-processing

Raw images go through two steps before being fed into the model:

**Square padding** — the hand region of interest (ROI) is unlikely to be perfectly square. Rather than stretching the image (which distorts hand shape), black padding is added to the shorter side to make it square whilst keeping the original proportions. Handled by `square_pad()` in `processing.py`.

**VGG normalisation** — the image is resized to 224x224 pixels (the input size expected by the pre-trained networks) and zero-centred using the ImageNet mean pixel values:
- Red: subtract 123.68
- Green: subtract 116.779
- Blue: subtract 103.939

This normalisation matches what the pre-trained models saw during ImageNet training, which is important for transfer learning to work well.

For the Massey University dataset, the green channel was dropped entirely to remove green screen background artefacts — with no noticeable drop in performance.

# 3. Transfer learning & feature extraction

Training a deep neural network from scratch requires a huge dataset and significant compute. Transfer learning sidesteps this — we take a model already trained on ImageNet (millions of images, 1000 classes) and repurpose its learned features for ASL classification.

The approach used here is **bottleneck feature extraction**:

1. Load a pre-trained network (VGG16, ResNet50, InceptionV3, Xception, or MobileNet) **without** its top classification layer
2. Pass all training images through this frozen network to produce a compact feature vector per image
3. Save those feature vectors to disk with `joblib`
4. Train only a small custom classification block on top of those saved features

This is much faster than training end-to-end — the heavy base model only needs to run once. Feature extraction was run on an AWS EC2 p2.8xlarge GPU instance.

The custom classification block:
```
Flatten → Dense(256, ReLU) → Dropout(0.2) → Dense(26, Softmax)
```
The final layer outputs a probability for each of the 26 ASL letters.

# 4. Training

Training is handled by `training_scripts/new_classifier.py`. The base model layers are frozen — only the custom classification block is trained.

Key training settings:
- **Optimiser:** Adadelta
- **Loss:** categorical cross-entropy
- **Epochs:** 30 (configurable)
- **Batch size:** 32 (configurable)
- **Data augmentation:** random rotation (±15°), width/height shifts (±15%) — helps the model generalise to different hand positions and angles
- **Callbacks:** ModelCheckpoint saves the best weights by validation accuracy

Training and validation loss/accuracy are saved as `.npy` files for later analysis.

# 5. Real-time system

The live demo (`live_demo.py`) ties everything together:

```
Webcam → Capture frame → Crop ROI → Square pad → Normalise → Model → Display top 3 predictions
```

1. **Webcam capture** — OpenCV reads frames continuously from the default camera
2. **ROI** — a fixed rectangle is drawn on screen; place your hand inside it. This region is cropped from each frame
3. **Pre-processing** — the cropped hand image is padded, resized to 224x224, and normalised (see Section 2)
4. **Prediction** — the full model (base network + classification block) runs inference on the processed frame
5. **Display** — if the top prediction confidence exceeds 50%, the top 3 predicted letters are overlaid on the live feed. The most probable letter is shown large in the centre; 2nd and 3rd choices appear smaller at the bottom

The 50% confidence threshold filters out low-quality frames (no hand present, partially obscured) to avoid noisy predictions.

# 6. References
https://research.gallaudet.edu/Publications/ASL_Users.pdf
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

