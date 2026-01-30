import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def crop_face(image):
    """
    Takes a BGR image
    Returns list of cropped face images
    """

    if image is None:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    results = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        if face.size != 0:
            results.append((face, (x, y, w, h)))

    return results
