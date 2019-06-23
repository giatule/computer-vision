# Môn học: Chuyên đề nghiên cứu về Thị giác máy tính (CS2309)

### Giảng viên: TS. Lê Đình Duy

### Học viên: Lê Gia Tự

### Mã số: CH1702048
# Bài tập số 2: Triển khai bài toán nhận diện gương mặt (Face detection)
# Nội dung demo:



## Getting Started

Face recognition là ứng dụng nhận diện khuông mặt tuyệt vời. Trên thực tế viết một ứng dụng nhận diện khuôn mặt từ một hình ảnh bất kỳ đã trở nên đơn giản với việc ứng dụng các thư viện có sẵn.
ageitgey/face_recognition là một trong những thư viện đáp ứng chức tốt nhất vơi độ chính xác cao và đã đạt được hơn 24000 star trên github.

Bài thực hành này sử dung thuật toán nhận diện khuôn mặt KNN - K-Nearest-Neighbors 

Ví dụ này hữu ích khi bạn muốn nhận ra một nhóm lớn những người đã biết và đưa ra dự đoán cho một người chưa biết trong thời gian tính toán khả thi.

### Mô Tả thuật toán:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under eucledian distance)
in its training set, and performing a majority vote (possibly weighted) on their label.
For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.

Trình phân loaị KNN được training từ một bộ dữ liệu những khuôn mặt đã được đánh nhãn và có thể tiên đoán được người trong hình ảnh chưa xác định bằng cách tìm k các khuôn mặt tương tự (dựa trên đặc tính khoảng cách của các chi tiết khuôn mặt như mắt, muổi, miệng) từ bộ training, và thực hiện việc bình chọn với trọng số khả thi nhất với nhãn.

Ví dụ: 
nếu k = 3 và ba hình ảnh khuôn mặt gần nhất với hình ảnh đã cho trong tập huấn luyện là một hình ảnh của Biden và hai hình ảnh của Obama, Kết quả sẽ là 'Obama'.

* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.
Usage:
Việc thực hiện việc bầu chọn có trọng số 
ứng dụng:
1. Prepare a set of images of the known people you want to recognize. Organize the images in a single directory
   with a sub-directory for each known person.
1. Chuẩn bị bộ dataset là khuôn mặt những người đã biết mà đã thực nghiệm ở bài lab số 1.Các khuôn mặt được lưu trữ trong folder con là tên của người đó.
2. Then, call the 'train' function with the appropriate parameters. Make sure to pass in the 'model_save_path' if you
   want to save the model to disk so you can re-use the model without having to re-train it.
   
   Sau đó gọi chức năng tr
3. Call 'predict' and pass in your trained model to recognize the people in an unknown image.


### Prerequisites

Bài lab thực nghiệm với python 3.7 chạy với hệ điều hành Mac OS 10.14 sử dụng editor Visual StudioCode (VScode)
đã cài các thư viện: dlib, cmake, face_recognition

Cú pháp cài đặt như sau: 
```
pip install dlib
pip install cmake
pip install face_recognition
```
### Thực nghiệm: 
![](https://github.com/giatule/computer-vision/blob/master/readme_img/face_recognition.gif)
