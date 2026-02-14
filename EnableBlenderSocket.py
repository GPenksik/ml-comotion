import bpy
import socket
import threading

def start_socket_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("localhost", 9999))
    server.listen(1)
    print("Socket server started on port 9999")

    while True:
        conn, addr = server.accept()
        print(f"Connection from {addr}")
        data = conn.recv(1024).decode("utf-8")
        exec(data)  # Execute the received Python code
        conn.close()

threading.Thread(target=start_socket_server, daemon=True).start()