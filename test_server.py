

from server.server_thread import ServerTask
import signal
import time

stopped = False
server = None
# Define the signal handler function
def signal_handler(sig, frame):

    server.stop()
    server.join()

    print("Stopped all servers!!!")
    exit(0)  # Exit the program

# Register the signal handler for SIGINT (Ctrl + C)
signal.signal(signal.SIGINT, signal_handler)

def main():
    """main function"""
    global server
    server = ServerTask()
    server.start()

    try:
        while not stopped:
            time.sleep(1)
    except Exception as e:
        raise
    except (KeyboardInterrupt, SystemExit):
        print(" Program is exiting by SIGINT signal...")
        pass
if __name__ == "__main__":
    main()