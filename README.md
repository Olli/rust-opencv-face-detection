# Face detection

This simple app is a Rust based face detection using OpenCV bindings of Rust.
It shows a rectangle around the detected face, a confidence level and landmarks.
The face is detected with a [yunet model](https://huggingface.co/opencv/face_detection_yunet/).
This whole program is basically only the CPP demo from OpenCV zoo but in Rust
For fast and easy results just 
```
cargo run
```
