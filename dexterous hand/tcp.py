import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(10)
s.connect(("192.168.5.100", 6601))     # 连手臂的 6601

s.sendall(b"hello arm\n")              # 发一句问候
print("arm says:", s.recv(1024).decode())   # 等它回话

s.close()