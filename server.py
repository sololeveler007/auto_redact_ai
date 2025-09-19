from flask import Flask, request, render_template, send_file, session
from io import BytesIO
from datetime import datetime
from processing_handler import handle_input
from flask_socketio import join_room, rooms
import uuid
from shared import bgScheduler, socketio, bytes_storage, log_data, event_queue,handle_emitting_messages
import os

app = Flask(__name__) #* flask setup
app.secret_key = "3452895464"
socketio.init_app(app, async_mode="eventlet")
bgScheduler.start() #* start the scheduler
folder = "./temp_files" 

@app.route('/')
@app.route('/index')

def home():
    if "user_id" not in session:
        user_id = str(uuid.uuid4())
        session['user_id'] = user_id
        session['log_data'] = {}
        # log_data[user_id] = {}
    user_id = session.get("user_id")
    log_data[user_id] = {}
    # print(log_data)
    return render_template('temp_index.html')    

@socketio.on("connect")
def handle_socketio_connect():
    user_id = session.get("user_id")
    if user_id:
        join_room(user_id) #* add the new_tab(connection) to the user room

@socketio.on("file_submitted")
def start_task(data):
    socketio.sleep(1)
    socketio.start_background_task(handle_emitting_messages, session.get("user_id")) #* start the bg task that sends completed message

@app.route('/download', methods =["POST"])
def img_process():
    user_id = session.get("user_id")
    
    if request.method == "POST":
        global folder
        input_time = datetime.now()
        received_file = request.files['sfile'] #* get the file from request
        uploaded_file_name = received_file.filename
        log_data[user_id]['mime_type'] = received_file.mimetype #* set mimetype 
        log_data[user_id]['uploaded_file_name'] = uploaded_file_name #* set uploaded_file_name
        unique_file_name = f"{uuid.uuid4()}" + '_' + uploaded_file_name
        input_file_path = os.path.join(f"./temp_files",unique_file_name).replace('\\','/')
        output_file_path = os.path.join("./temp_files_processed",unique_file_name).replace('\\','/')
        log_data[user_id]['input_file_path']  = input_file_path
        log_data[user_id]['output_file_path']  = output_file_path
        received_file.save(input_file_path)
        event_queue.put(("input_file_processing", uploaded_file_name)) #* add a job to queue
        bgScheduler.add_job(handle_input, args=[input_file_path, input_time, output_file_path, user_id]) #* start the background job
        
        # return render_template('sample_out_web.html', file_name = uploaded_file_name)
        return render_template('temp_output_page.html', file_name = uploaded_file_name)
       
    return "File was not uploaded"

@app.route('/download', methods = ['GET'])
def sending_processed_image_for_download():
    user_id = session.get("user_id")
    return send_file(
        log_data[user_id]['output_file_path'],
        mimetype = log_data[user_id]['mime_type'],
        as_attachment = True,
        download_name= f"processed{log_data[user_id]['file_type']}"
    )

if __name__ == '__main__':
    socketio.run(app, host= '0.0.0.0', port=8000)