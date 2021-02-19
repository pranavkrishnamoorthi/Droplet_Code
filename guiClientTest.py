# import http.client

# c = http.client.HTTPConnection('localhost', 8080)
# c.request('POST', '/process', '{}')
# doc = c.getresponse().read()
# print(doc)
# # 'All done'

from threading import Thread, Event
import tkinter as tk

msg = 'hi'


def worker(run_event):
    global msg
    while run_event.is_set():
        print(msg)
        if msg == 'q':
            print("ending")
            print("eending")
            print('eee')


window = tk.Tk()
label = tk.Label(text="Enter a radius value")
entry = tk.Entry()
button = tk.Button(text="Set Radius")
buttonQuit = tk.Button(text="Exit Controller")


def radiusClick(event):
    global msg
    try:
        msg = int(entry.get())
    except:
        return


def quitClick(event):
    global window
    window.quit()


window.geometry("500x200")
button.bind('<Button>', radiusClick)
buttonQuit.bind('<Button>', quitClick)
label.pack()
entry.pack()
button.pack()
buttonQuit.pack()


if __name__ == '__main__':
    run_event = Event()
    run_event.set()
    t1 = Thread(target=worker, args=(run_event,))
    t1.daemon = True
    t1.start()
    # while msg != 'q':
    #     try:
    #         msg_raw = input()
    #         if msg_raw == 'q':
    #             run_event.clear()
    #             print("normal")
    #             break
    #         try:
    #             msg = int(msg_raw)
    #         except:
    #             continue
    #     except:
    #         msg = 'q'
    #         run_event.clear()
    #         print("hehe")
    #         break

    try:
        window.mainloop()
        run_event.clear()
    except:
        run_event.clear()
        print("hehe")
    run_event.clear()
