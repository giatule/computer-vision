# Chuyên đề nghiên cứu về Thị giác máy tính (CS2309)
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
```
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.
    :param train_dir: directory that contains a sub-directory for each known person, with its name.
     (View in source code to see train_dir example tree structure)
     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(img_path, predictions):
    """
    Shows the face recognition results visually.
    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()


if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    print("Training KNN classifier...")
    classifier = train("knn_data/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")

    # STEP 2: Using the trained classifier, make predictions for unknown images
    for image_file in os.listdir("knn_data/test"):
        full_file_path = os.path.join("knn_data/test", image_file)

        print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(full_file_path, model_path="trained_knn_model.clf")

        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))

        # Display results overlaid on an image
        show_prediction_labels_on_image(os.path.join("knn_data/test", image_file), predictions)
```
![](https://github.com/giatule/computer-vision/blob/master/readme_img/check_exist_face.png)
![](https://github.com/giatule/computer-vision/blob/master/readme_img/face_recognition.gif)
