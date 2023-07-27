import face_recognition

stored_faces = []
faces_names = []


def store_current_face(face, name):
    try:
        face_encode = face_recognition.face_encodings(face)[0]
        stored_faces.append(face_encode)
        faces_names.append(name)
    except IndexError:
        pass#print("No Face")
    return name


def store_face_name_with_encoding(face_encode, name):
    try:
        stored_faces.append(face_encode)
        faces_names.append(name)
    except IndexError:
        pass#print("No Face")
    return name


def recognize_face(face_rectangle):
    try:
        face_encode = face_recognition.face_encodings(face_rectangle)[0]
        result = face_recognition.compare_faces(stored_faces, face_encode)
        print(result)
        index=[i for i,x in enumerate(result) if x]
        return faces_names[index[0]]
    except IndexError:
        pass#print("No Face")


def return_face_by_name(name):
    i = faces_names.index(name)
    return stored_faces[i]


def delete_face_by_name(name):
    i = faces_names.index(name)
    faces_names.pop(i)
    stored_faces.pop(i)
