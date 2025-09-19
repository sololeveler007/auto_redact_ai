from apscheduler.schedulers.background import BackgroundScheduler
from flask_socketio import SocketIO
from io import BytesIO
from queue import Queue
import os

socketio = SocketIO() #* create a SocketIO instance that gets imported and initialized
bgScheduler = BackgroundScheduler() #* create background scheduler instance
bytes_storage = BytesIO()
log_data = dict()
event_queue = Queue()

def handle_emitting_messages(user_id):
    acknowledged = False
    event_queue.join() #* wait until the job is done
    process_name, file_name = event_queue.get() #* get job details

    def change_value(data):  #* acknowledgement confirmation from frontend
        nonlocal acknowledged
        acknowledged = True
        socketio.start_background_task(delete_files, user_id=user_id) #* start task to delete files

    def try_emit(): #* emit message
        nonlocal acknowledged
        if not acknowledged:
            socketio.emit("input_file_processed", {"message":"completed", "current_file_instance": log_data[user_id]["uploaded_file_name"]}, to=user_id, callback=change_value)
            socketio.start_background_task(delete_files, user_id=user_id) #* start task to delete files
    try_emit()

def delete_files(user_id): #* func to  delete files after acknowledgement is received
    global log_data
    input_file_path = log_data[user_id]['input_file_path']
    output_file_path = log_data[user_id]['output_file_path']
    socketio.sleep(100)
    if os.path.exists(input_file_path) and os.path.exists(output_file_path):
        os.remove(input_file_path)
        os.remove(output_file_path)
        socketio.emit("files_deleted", {"message":"input_output_files_deleted", "deleted_file_instance": log_data[user_id]["uploaded_file_name"]}, to=user_id)


if __name__ == "__main__":
    pass