# Realtime-Handsign-to-Audio-Converter
This is a project which can detect alphabets and certain words of ASL(American Sign Language) and give audio output in real time.

## Instructions to use this repo
* Open the folder in any IDE or VS Code Text Editor and run the hand-sign-recognizer_copy.py
* Run this command in Terminal "pip install pickle-mixin opencv-python mediapipe numpy tkinter pyttsx3"(make sure you have selected the correct Python Environment)
* Make sure you have all the libraries installed in your environment.
* Required libraries are
  1. pickle: The Python standard library includes pickle, so no need to install it separately.
  2. opencv-python: For cv2.
  3. mediapipe: For MediaPipe functionality.
  4. numpy: For numerical computations.
  5. tkinter: Included with Python's standard library; no installation required unless you're using a minimal Python distribution.
  6. pyttsx3: For text-to-speech functionality.
* Run the file and show a hand sign.
* Wait for 3-5 seconds and if the accuracy of the sign is more than 50% for more than 10 frames, it will give an audio output.(this can be altered inside the code.) 



## Important 
* The only signs these models can detect are given in label files(refer these files for words).
* The words "Fine" and "Mother/Father" have low accuracy because the hand points cannot be detected Properly.
* The letters "J" and "Z" are not included because they have motion in them and these models are trained using static images.
* Once an alphabet or a word is detected and spoken, it will not give the audio output again of the prediction. This has been done to stop the repetitive audio output of the same prediction.
