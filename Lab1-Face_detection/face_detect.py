# đếm số lượng khuôn mặt xuất hiện trong hình
import face_recognition
Img_in_path="Input/lab2.jpg"
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

    # Bạn có thể truy cập vào khuôn mặt thực tế như thế này:
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()
