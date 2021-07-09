# Automated Class Attendance with Python & OpenCV
Automated class attendance system built with OpenCV to perform multi-person detection & recognition of students

The aim of this project was to implement a camera based solution that periodically takes a still image from a webcam and detects the faces
of students. The system then recognises the students present and automatically updates a class register database with attendance and 
information on latecoming.

However, the implementation that was achieved performed multi-face recognition in real time and continuously updated the recognition model by periodically retraining it with up-to-date pictures of each correctly identified student.

![Mockup of what the recognition looked like](http://i.imgur.com/5swfQgw.png?1)
