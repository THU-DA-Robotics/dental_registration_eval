import socket

def send_cmd(cmd):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(10)
    s.connect(("192.168.5.100", 6601))
    s.sendall((cmd + "\n").encode())
    reply = s.recv(1024).decode().strip()
    s.close()
    return reply

# 1. 先确认回声还通
print("echo:", send_cmd("hello"))

# 2. 使能（如果还没使能）
print("enable:", send_cmd("Enable"))

# 3. 压速度到 20%（慢就是安全）
print("speed:", send_cmd("Speed 20"))

# 4. 移动到安全点（把下面的坐标换成你刚才抄的！）
# 格式：MovJ X,Y,Z,Rx,Ry,Rz
print("move:", send_cmd("MovJ 200,0,100,0,0,0"))   # <--- 改成你的坐标！

# 5. 再移到第二个点（验证它真的在动）
print("move2:", send_cmd("MovJ 250,0,100,0,0,0"))  # <--- 改成你的第二个坐标！