import socket
from pynput import keyboard

arm = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
arm.connect(("192.168.5.100", 6601))

def arm_cmd(c):
    arm.sendall((c + "\n").encode())
    print("arm:", arm.recv(64).decode().strip())

def on_press(key):
    try:
        c = key.char.lower()
    except AttributeError:
        return
    if c in "wsadrf":
        arm_cmd(c)
    elif c == "q":
        return False

print("W/S=前后  A/D=左右  R/F=上下  Q=退出")
with keyboard.Listener(on_press=on_press) as l:
    l.join()