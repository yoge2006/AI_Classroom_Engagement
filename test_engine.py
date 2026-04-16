import sys
import os
import threading
import traceback
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.attention.attention_engine import run_attention_engine

def callback(data):
    pass

def start_ai():
    try:
        print("Starting AI engine...")
        run_attention_engine(callback)
    except Exception as e:
        print("Error caught in start_ai:")
        traceback.print_exc()

if __name__ == "__main__":
    t = threading.Thread(target=start_ai)
    t.start()
    
    while t.is_alive():
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping...")
            break
    print("Done")
