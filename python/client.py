import socket
import time
import json
import statistics

def venus(ip, port, meld):
    network_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    network_socket.settimeout(3)
    time_list = []
    try:
        network_socket.connect((ip, port))
    except Exception as e:
        print(e)
        print("Could not connect to network")
        exit()
    for __ in range(100):
        try:
            start = time.time_ns()
            network_socket.sendall(str.encode(meld))
            recmeld = network_socket.recv(1024)
            print(time_list.append(time.time_ns()))
        except Exception as e:
            print(e)
            print("Connection lost")
            break
        while True:
            melding = network_socket.recv(1024)
            if len(melding) > 1:
                break
        #print(melding)
        tid = time.time_ns()-start
        time_list.append(tid)
        #recmeld = network_socket.recv(1024)
    time.sleep(1)
    network_socket.close()
    print (f'Mean:{statistics.mean(time_list)}\n Max:{max(time_list)}')
    return "ok"

if __name__ == "__main__":
    print("Main=client")
    dictionary = {"can":[(59,"datadata")]}
    meld = json.dumps(dictionary)
    ip = "10.0.0.2"
    port = 6900
    svar = print(venus(ip, port, meld))
    