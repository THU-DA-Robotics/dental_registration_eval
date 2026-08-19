import socket

def send(sock, cmd):
    sock.sendall((cmd + "\n").encode())
    reply = sock.recv(1024).decode()
    print(f"我喊: {cmd}   它答: {reply.strip()}")
    return reply

dash = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
dash.connect(("192.168.5.100", 29999))

send(dash, "RobotMode()")
send(dash, "EnableRobot()")