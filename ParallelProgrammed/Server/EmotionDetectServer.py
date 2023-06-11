import socket

def server_program():
    host = "192.168.43.180"
    port = 2048

    server_socket=socket.socket()
    server_socket.bind((host,port))

    server_socket.listen(1)

    conn, address = server_socket.accept()
    print("Connection from: " + str(address))

    while True:

        data = conn.recv(1024).decode()
        print ("From connected user: " + str(data))
        data = input(' -> ')
        conn.send(data.encode())

    conn.close()

if __name__== '__main__':
    server_program()
