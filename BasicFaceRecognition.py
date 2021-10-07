import cv2
import numpy as np
import face_recognition

imgImdad = face_recognition.load_image_file('images/Imdad4.jpg')
imgImdad = cv2.cvtColor(imgImdad, cv2.COLOR_BGR2RGB)
imgImdadTest = face_recognition.load_image_file('images/Imdad6.jpg')
imgImdadTest = cv2.cvtColor(imgImdadTest, cv2.COLOR_BGR2RGB)

# Make rectangle for Input Dataset
faceLoc = face_recognition.face_locations(imgImdad)[0]
encodeImdad = face_recognition.face_encodings(imgImdad)[0]
cv2.rectangle(imgImdad, (faceLoc[3], faceLoc[0]), (faceLoc [1], faceLoc[2]), (255,0,255), 2)

# Make rectangle for Test Dataset
faceLocTest = face_recognition.face_locations(imgImdadTest)[0]
encodeImdadTest = face_recognition.face_encodings(imgImdadTest)[0]
cv2.rectangle(imgImdadTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest [1], faceLocTest[2]), (255,0,255), 2)

# Show the matching result
results = face_recognition.compare_faces([encodeImdad], encodeImdadTest)
# print(results)
faceDistance = face_recognition.face_distance([encodeImdad], encodeImdadTest)
print(results, faceDistance)
cv2.putText(imgImdadTest, f'{results} {round(faceDistance[0], 2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

cv2.imshow("Imdadul Haque-Musk", imgImdad)
cv2.imshow("Imdadul Haque-Test", imgImdadTest)
cv2.waitKey(0)