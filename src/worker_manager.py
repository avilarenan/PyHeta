import subprocess
import sys, os
from time import sleep
import sys
import signal
import argparse
import psutil
parser = argparse.ArgumentParser("simple_example")
parser.add_argument("n_workers", help="Number of workers to launch.", type=int)
parser.add_argument("worker_file", help="Python file to launch process.", type=str)
args = parser.parse_args()

N_WORKERS = args.n_workers
WORKER_FILE = args.worker_file

processes = []

def gracefully_shutdown():
    for process in processes:
        process.kill()
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)

def signal_term_handler(signal, frame):
    print("got SIGTERM")
    gracefully_shutdown()
 
signal.signal(signal.SIGTERM, signal_term_handler)

if __name__ == '__main__':
    try:
        for i in range(N_WORKERS):
            print("Creating process")
            process = subprocess.Popen(["python3", WORKER_FILE])
            print(f"Created process {process.pid}")
            processes += [process]
        while True:

            procs_to_remove = []

            for proc in processes:
                if proc.poll() is not None:
                    procs_to_remove += [proc]

            for proc in procs_to_remove:
                processes.remove(proc)
                proc.kill()
                process = subprocess.Popen(["python3", WORKER_FILE])
                processes += [process]
                
            sleep(60)

    except KeyboardInterrupt:
        print('Interrupted - Gracefully killing')
        for process in processes:
            process.kill()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)