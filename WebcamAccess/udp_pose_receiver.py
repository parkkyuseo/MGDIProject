# udp_pose_receiver.py (Windows)
import socket, json, time, sys
import msvcrt  # Windows-only

PORT = 19561  # HoloLens 스크립트의 pcPort와 맞춰야 함

def key_pressed():
    if msvcrt.kbhit():
        ch = msvcrt.getch()
        if not ch:
            return None
        try:
            c = ch.decode('utf-8').lower()
        except:
            c = ''
        return c
    return None

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    sock.settimeout(0.2)
    print(f"listening on UDP {PORT} ...  (press 'q' or ESC to quit)")
    count = 0
    last_from = None
    last_image = None
    while True:
        # 키 체크
        c = key_pressed()
        if c in ('q', '\x1b'):  # q or ESC
            print("bye.")
            break

        try:
            data, addr = sock.recvfrom(65535)
            last_from = addr
            count += 1
            msg = json.loads(data.decode('utf-8'))
            pose = msg.get('data', msg)  # Wrapper 대응
            last_image = pose.get('image')
            print(f"[{count}] from {addr}  image={last_image}  t_HC={pose.get('t_HC')}")
        except socket.timeout:
            pass
        except Exception as e:
            print("bad packet:", e)

        # 과도한 로그 방지용 짧은 슬립
        time.sleep(0.01)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("bye.")
