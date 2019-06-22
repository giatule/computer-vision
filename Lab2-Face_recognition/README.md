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
1. Chuẩn bị bộ dataset là khuôn mặt những người đã biết mà đã thực nghiệm ở bài lab số 1.
2. Then, call the 'train' function with the appropriate parameters. Make sure to pass in the 'model_save_path' if you
   want to save the model to disk so you can re-use the model without having to re-train it.
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
### Mô Tả Thuật Toán KNN

### Installing
### Step 1
Make a folder called ./training-images/ inside the openface folder.

mkdir training-images
### Step 2
Make a subfolder for each person you want to recognize. For example:

mkdir ./training-images/will-ferrell/
mkdir ./training-images/chad-smith/
mkdir ./training-images/jimmy-fallon/
### Step 3
Copy all your images of each person into the correct sub-folders. Make sure only one face appears in each image. There's no need to crop the image around the face. OpenFace will do that automatically.

### Step 4
Run the openface scripts from inside the openface root directory:

First, do pose detection and alignment:

./util/align-dlib.py ./training-images/ align outerEyesAndNose ./aligned-images/ --size 96

This will create a new ./aligned-images/ subfolder with a cropped and aligned version of each of your test images.

Second, generate the representations from the aligned images:

./batch-represent/main.lua -outDir ./generated-embeddings/ -data ./aligned-images/

After you run this, the ./generated-embeddings/ sub-folder will contain a csv file with the embeddings for each image.

Third, train your face detection model:

./demos/classifier.py train ./generated-embeddings/

This will generate a new file called ./generated-embeddings/classifier.pkl. This file has the SVM model you'll use to recognize new faces.

At this point, you should have a working face recognizer!

### Step 5: Recognize faces!
Get a new picture with an unknown face. Pass it to the classifier script like this:

./demos/classifier.py infer ./generated-embeddings/classifier.pkl your_test_image.jpg

You should get a prediction that looks like this:

=== /test-images/will-ferrel-1.jpg ===
Predict will-ferrell with 0.73 confidence.
From here it's up to you to adapt the ./demos/classifier.py python script to work however you want.

### Important notes:

If you get bad results, try adding a few more pictures of each person in Step 3 (especially picures in different poses).
This script will always make a prediction even if the face isn't one it knows. In a real application, you would look at the confidence score and throw away predictions with a low confidence since they are most likely wrong.
