# Detect Face Masks in Videos/GIFs

**Version 1.0.0**

Check whether people appearing in videos or GIFs are wearing face masks or not.

---

## Project Description

This project uses the face_recognition library to detect the locations of faces.

Encodings are being saved as well and uniques ids being assigned to every face. The ids are stored in the folder "known_faces" and are deleted at the end of the program.

The ids can be used to recognise faces if facial recognition is needed.

If specific faces are to be recognised, add a folder for every face inside the "known_faces" folder. The faces will be recognised by the folder names.

Remove the lines giving unique id to faces and use the names of the folders instead.

Notebook for creating the classification model will be uploaded soon.

## Project Setup

1. Clone the project using the following command: 
```bash
git clone https://github.com/DiamondGible2/Face-Masks-in-Videos.git 
```
2. Install the necessary libraries. The links are given below can be used for the same.
3. cd into the project folder
4. Run the file and the video will be displayed with the results on it.
5. To change the video file, open the face_mask_check.py file and the name inside cv2.VideoCapture()

🌟 You are all set!

## Relevant Links

* [face_recognition GitHub Repository](https://github.com/ageitgey/face_recognition)
* [face_recognition Installation page](https://pypi.org/project/face-recognition/)
* [TensorFlow Installation Page](https://www.tensorflow.org/install)
* [OpenCv Installation Page](https://pypi.org/project/opencv-python/)
* [Python Download Page](https://www.python.org/downloads/)