# Project Title: 

Bài lab thực nghiệm về nhận diện khuông mặt trong Python

## Getting Started

Face recognition là ứng dụng nhận diện khuông mặt tuyệt vời. Trên thực tế viết một ứng dụng nhận diện khuôn mặt từ một hình ảnh bất kỳ đã trở nên đơn giản với việc ứng dụng các thư viện có sẵn.
ageitgey/face_recognition là một trong những thư viện đáp ứng chức tốt nhất vơi độ chính xác cao và đã đạt được hơn 24000 star trên github.


### Prerequisites

Bài lab thực nghiệm với python 3.7 chạy với hệ điều hành Mac OS 10.14
Vitual Studial Code (VScode)
đã cài các thư viện: dlib, cmake, face_recognition

Cú pháp cài đặt như sau: 
```
pip install dlib
pip install cmake
pip install face_recognition
```

### Installing

A Simple Example
```
import face_recognition

image = face_recognition.load_image_file("My_Image.png")
face_locations = face_recognition.face_locations(image)
print("I found {} face(s) in this photograph.".format(len(face_locations)))
```



```

```




