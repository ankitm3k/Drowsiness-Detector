# Drowsiness-Detector

This project aims to detect drowsiness in a person using the Eye Aspect Ratio (EAR) technique. It uses facial landmarks detection to locate the eyes and calculate the EAR value, which is a measure of the eyelid openness. If the EAR value goes below a certain threshold, the program will issue a warning to alert the user of possible drowsiness.

## Requirements

Download and paste the `shape_predictor_68_face_landmarks.dat` file from this [link](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2) into the project folder.

You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

To run the program with a webcam, execute the following command:

```bash
python drowsiness_detector.py --webcam
```

To run the program with a video file, execute the following command:

```bash
python drowsiness_detector.py --video /path/to/video
```

Use 'q' key to quit the program.

## How it works

The program detects faces in the frame using the dlib frontal face detector. For each face, the facial landmarks are detected using the dlib facial landmarks detector. The landmarks corresponding to the left and right eyes are identified, and the EAR value is calculated using the Euclidean distance between the landmarks.

If the EAR value goes below a certain threshold, the program will issue a warning indicating possible drowsiness.

The program uses multithreading to improve the performance. The process_frame() function is executed in a separate thread for each frame of the video stream.


## Credits

* The face detection and landmark detection models used in this project are provided by the dlib library, which can be found at [link](http://dlib.net/).
* The eye aspect ratio calculation function was inspired by the work of Tereza Soukupova and Jan Cech in their paper, "Real-Time Eye Blink Detection using Facial Landmarks," which can be found at [link](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf).
