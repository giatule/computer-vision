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
