# import http.client

# c = http.client.HTTPConnection('localhost', 8080)
# c.request('POST', '/process', '{}')
# doc = c.getresponse().read()
# print(doc)
# # 'All done'

from threading import Thread, Event

msg = 'hi'


def worker(run_event):
    global msg
    while run_event.is_set():
        print(msg)
        if msg == 'q':
            print("ending")
            print("eending")
            print('eee')


if __name__ == '__main__':
    run_event = Event()
    run_event.set()
    t1 = Thread(target=worker, args=(run_event,))
    t1.daemon = True
    t1.start()
    while msg != 'q':
        try:
            msg_raw = input()
            if msg_raw == 'q':
                run_event.clear()
                print("normal")
                break
            try:
                msg = int(msg_raw)
            except:
                continue
        except:
            msg = 'q'
            run_event.clear()
            print("hehe")
            break
