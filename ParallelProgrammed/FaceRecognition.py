import face_recognition

stored_faces = []
faces_names = []


def store_current_face(face, name):
    try:
        face_encode = face_recognition.face_encodings(face)[0]
        stored_faces.append(face_encode)
        faces_names.append(name)
    except IndexError:
        print("No Face")
    return name


def store_face_name_with_encoding(face_encode, name):
    try:
        stored_faces.append(face_encode)
        faces_names.append(name)
    except IndexError:
        print("No Face")
    return name


def recognize_face(face_rectangle):
    try:
        face_encode = face_recognition.face_encodings(face_rectangle)[0]
        for i in range(len(stored_faces)):
            result = face_recognition.compare_faces([stored_faces[i]], face_encode)
            if result:
                return faces_names[i]
    except IndexError:
        print("No Face")


def return_face_by_name(name):
    i = faces_names.index(name)
    return stored_faces[i]


def delete_face_by_name(name):
    i = faces_names.index(name)
    faces_names.pop(i)
    stored_faces.pop(i)
