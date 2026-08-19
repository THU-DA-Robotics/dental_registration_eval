import socket

for port in [29999, 30003, 30004, 6600, 6601, 502, 8080, 22]
    s = socket.socket()
    s.settimeout(0.5)
    try
        s.connect((192.168.5.100, port))
        print(port, == 开着！)
    except Exception
        print(port, 关着)
    finally
        s.close()