import argparse
import io
from PIL import Image
import datetime
import torch
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, Response, send_from_directory
from werkzeug.utils import secure_filename
import os
import time
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'runs/detect'
app.config['STATIC_FOLDER'] = 'static/runs/detect'

# Ensure the upload and output folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
            f.save(filepath)
            file_extension = f.filename.rsplit('.', 1)[1].lower()

            if file_extension in ['jpg', 'jpeg', 'png']:
                img = cv2.imread(filepath)
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                # Perform the detection
                yolo = YOLO('best.pt')
                results = yolo.predict(source=image, save=True, project=app.config['OUTPUT_FOLDER'], name='predict')

                # Print the results to debug
                print(f"Results: {results}")

                # Get the saved image path from the results
                saved_image_path = results[0].save_dir / results[0].name

                # Construct the full path
                output_image_path = os.path.join(saved_image_path, os.path.basename(filepath))
                static_output_image_path = os.path.join(app.config['STATIC_FOLDER'], 'predict', os.path.basename(filepath))

                # Print the paths for debugging
                print(f"Output image path: {output_image_path}")
                print(f"Static output image path: {static_output_image_path}")

                # Move the processed image to the static folder for serving
                os.makedirs(os.path.dirname(static_output_image_path), exist_ok=True)
                try:
                    os.rename(output_image_path, static_output_image_path)
                except FileNotFoundError as e:
                    print(f"FileNotFoundError: {e}")
                    return "File not found. Please check the file paths and ensure the detection process completed successfully.", 500

                return render_template('index.html', image_path=static_output_image_path)

            elif file_extension == 'mp4':
                video_path = filepath
                cap = cv2.VideoCapture(video_path)

                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.mp4')
                out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

                # Initialize YOLO model
                model = YOLO('best.pt')

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Do YOLO detection on the frame
                    results = model(frame)
                    res_plotted = results[0].plot()

                    # Write the frame to the output video
                    out.write(res_plotted)

                cap.release()
                out.release()

                return video_feed()

    return render_template('index.html')

def get_frame():
    video = cv2.VideoCapture(os.path.join(app.config['OUTPUT_FOLDER'], 'output.mp4'))
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)


@app.route("/video_feed")
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/webcam_feed")
def webcam_feed():
    cap = cv2.VideoCapture(0)

    def generate():
        while True:
            success, frame = cap.read()
            if not success:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            img = Image.open(io.BytesIO(frame))

            model = YOLO('best.pt')
            results = model(img)

            res_plotted = results[0].plot()

            img_BGR = cv2.cvtColor(np.array(res_plotted), cv2.COLOR_RGB2BGR)
            frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov8 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    model = YOLO('best.pt')
    app.run(host="0.0.0.0", port=args.port)
