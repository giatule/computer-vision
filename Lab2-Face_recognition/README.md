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
Ứng dụng:

1. Chuẩn bị bộ dataset là khuôn mặt những người đã biết mà đã thực nghiệm ở bài lab số 1.Các khuôn mặt được lưu trữ trong folder con là tên của người đó.

2.Sau đó, gọi hàm 'train' với các tham số thích hợp. Đảm bảo vượt qua trong 'model_save_path' nếu bạn muốn lưu mô hình vào đĩa để bạn có thể sử dụng lại mô hình mà không phải đào tạo lại mô hình.
  
3. Gọi hàm Predict để tiên đoán và chuyển qua mô hình được đào tạo của bạn để nhận ra những người trong một hình ảnh không xác định.


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
LAB 1
https://github.com/giatule/computer-vision/tree/master/Lab1-Face_detection

# Demo 1: simple_face_detect.py

```
import face_recognition
#Gán đường dẫn hình cần xét
Img_in_path="Lab1-Face_detection/Input/lab2.jpg"
image = face_recognition.load_image_file(Img_in_path)
face_locations = face_recognition.face_locations(image)
print("Tìm thấy {} khuôn mặt(s) trong ảnh.".format(len(face_locations)))
```
# Hình xét 

![](https://github.com/giatule/computer-vision/blob/master/Lab1-Face_detection/Input/lab2.jpg)

Kết quả thực nghiệm

![](https://github.com/giatule/computer-vision/blob/master/readme_img/simple_face_detect.png)


# Demo 2: face_detect.py
https://github.com/giatule/computer-vision/blob/master/Lab1-Face_detection/face_detect.py

```
# đếm số lượng khuôn mặt xuất hiện trong hình
import face_recognition
Img_in_path="Lab1-Face_detection/Input/lab2.jpg"
image = face_recognition.load_image_file(Img_in_path)
face_locations = face_recognition.face_locations(image)
print("I found {} face(s) in this photograph.".format(len(face_locations)))

#Nhận diện khuông mặt
from PIL import Image
import face_recognition

# Tải tập tin jpg vào một mảng numpy
image = face_recognition.load_image_file(Img_in_path)

# Tìm tất cả các khuôn mặt trong ảnh bằng mô hình dựa trên HOG-based model.
# Phương pháp này khá chính xác, nhưng không chính xác như mô hình CNN và không tăng tốc GPU.
# See also: find_faces_in_picture_cnn.py
face_locations = face_recognition.face_locations(image)

print("I found {} face(s) in this photograph.".format(len(face_locations)))

for face_location in face_locations:

    # In vị trí của mỗi khuôn mặt trong hình ảnh này
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # Cách thức truy cập vào từng khuôn mặt:
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()
```

# Kết quả thực nghiệm nhận diện khuôn mặt
# Kết quả terminal console: 
```
I found 4 face(s) in this photograph.
A face is located at pixel location Top: 582, Left: 951, Bottom: 689, Right: 1059
A face is located at pixel location Top: 477, Left: 1164, Bottom: 632, Right: 1319
A face is located at pixel location Top: 582, Left: 617, Bottom: 689, Right: 724
A face is located at pixel location Top: 540, Left: 1515, Bottom: 669, Right: 1644
```

# Kết quả ứng dụng: 

![](https://github.com/giatule/computer-vision/blob/master/readme_img/Demo2_face_detect.png)


# Demo 3: Kiểm tra nếu một người có tồn tại trong ảnh.

```
import face_recognition
# Load in our hinh anh tham chieu cua Le gia Tu
known_image = face_recognition.load_image_file("Lab1-Face_detection/train/Le Gia Tu/21. Le Gia Tu.jpg")
# Load Load một hình nhóm người bất kỳ
unknown_image = face_recognition.load_image_file("Lab1-Face_detection/Input/lab2.jpg")

# tạo một bộ encoding tu hinh mẫu
Tu_encoding = face_recognition.face_encodings(known_image)[0]
# tạo encoding cho hình nhóm người bất kỳ (hình chưa xác định)
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# So sánh đối chiếu encoding mấu với bộ encoding chưa xác định nếu khớp --> true/fo
results = face_recognition.compare_faces([Tu_encoding], unknown_encoding)
# Print the results
print("Kết quả nhận diện ")
print(results)
```
# terminal console: true/fail

![](https://github.com/giatule/computer-vision/blob/master/readme_img/check_exist_face.png)
![](https://github.com/giatule/computer-vision/blob/master/readme_img/face_recognition.gif)
