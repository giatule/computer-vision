import face_recognition
Img_in_path="Lab1-Face_detection/Input/lab2.jpg"
image = face_recognition.load_image_file(Img_in_path)
face_locations = face_recognition.face_locations(image)
print("Tìm thấy {} khuôn mặt(s) trong ảnh.".format(len(face_locations)))