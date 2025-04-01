from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from vector_database import VectorDB
import cv2
import os
from PIL import Image
import base64
from io import BytesIO
import torch
from torchvision import transforms
#from googletrans import Translator
import shutil

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Lấy API key từ biến môi trường
QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
if not QDRANT_API_KEY:
    print("Cảnh báo: QDRANT_API_KEY không được cung cấp trong biến môi trường. Sử dụng giá trị mặc định.")

qdrant_manager = VectorDB(
    api='http://aienthusiasm:6333',
    timeout=200.0,
    api_key=QDRANT_API_KEY
)

# Định nghĩa hàm clear_directory trước khi sử dụng
def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

image_folder = os.path.join('static', 'temporary', 'images')
os.makedirs(image_folder, exist_ok=True)

# Xóa các file hình ảnh tạm thời khi khởi động
if os.path.exists(image_folder):
    clear_directory(image_folder)

@app.route('/')
def home():
    return render_template('newhome.html')

@app.route('/static/<path:filename>')
def custom_static(filename):
    return send_from_directory('static', filename)

@app.route('/search', methods=['POST'])
def search():
    query_text = request.form.get('query')

    if not query_text:
        return jsonify({'error': 'No query or files provided'}), 400

    return search_images(query_text)

def process_qdrant_results(qdrant_results):
    scenes = {}

    for result in qdrant_results:
        video_folder = result.payload['video_folder']
        frame_number = result.payload['frame_number']
        frame_idx = result.payload['frame_idx']
        pts_time = result.payload['pts_time']
        compressed = result.payload['compressed']

        # Tạo đường dẫn tương đối với static_url_path
        output_path = os.path.join('temporary', 'images', f"{video_folder}_{frame_number}.jpg").replace('\\', '/')
        full_path = os.path.join('static', output_path)
        qdrant_manager.decode_and_decompress_image(compressed, full_path)

        # Sử dụng URL tương đối với static_url_path
        relative_path = f"/static/{output_path}"
        
        scene_identifier = (video_folder, frame_number)
        if scene_identifier not in scenes:
            scenes[scene_identifier] = {
                'metadata': {
                    'video_folder': video_folder,
                    'frame_number': frame_number,
                    'frame_idx': frame_idx,
                    'pts_time': pts_time,
                    'frame_path': relative_path
                }
            }

    return scenes

@app.route('/search_images', methods=['GET', 'POST'])
def search_images(query_text=None):
    if request.method == 'GET':
        query_text = request.args.get('query')
    elif request.method == 'POST':
        query_text = request.form.get('query')

    if not query_text:
        return jsonify({'error': 'Query text are required'}), 400

    qdrant_results = qdrant_manager.query_dataset(query_text)
    if not qdrant_results:
        return jsonify({'error': 'No images found. Try a different query.'}), 404

    scenes = process_qdrant_results(qdrant_results)

    metadata_list = [scene['metadata'] for scene in scenes.values()]
    frame_paths = [scene['metadata']['frame_path'] for scene in scenes.values()]

    return jsonify({
        'frame_paths': frame_paths,
        'metadata_list': metadata_list,
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)