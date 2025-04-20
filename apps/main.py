from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from vector_database import VectorDB
import cv2
import os
from PIL import Image
import base64
from io import BytesIO
import torch
from torchvision import transforms
import shutil
from dotenv import load_dotenv

# Khởi tạo ứng dụng Flask
app = Flask(__name__, static_folder='static', static_url_path='/static')
# static_folder: Thư mục chứa file tĩnh (CSS, JS, hình ảnh)
# static_url_path: Đường dẫn URL cho file tĩnh

# Tải biến môi trường từ file .env
load_dotenv()

# Lấy API key từ biến môi trường
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
if not QDRANT_API_KEY:
    print("Cảnh báo: QDRANT_API_KEY không được cung cấp trong biến môi trường. Sử dụng giá trị mặc định.")

# Khởi tạo đối tượng VectorDB để tương tác với Qdrant
qdrant_manager = VectorDB(
    api='http://aienthusiasm:6333',  # URL của server Qdrant
    timeout=200.0,  # Thời gian chờ tối đa cho yêu cầu
    api_key=QDRANT_API_KEY  # API key từ biến môi trường
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

# Định nghĩa thư mục lưu hình ảnh tạm thời
image_folder = os.path.join('static', 'temporary', 'images')
os.makedirs(image_folder, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

# Xóa các file hình ảnh tạm thời khi khởi động
if os.path.exists(image_folder):
    clear_directory(image_folder)  # Xóa toàn bộ nội dung thư mục
    
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

# Route xử lý tìm kiếm hình ảnh (GET hoặc POST)
@app.route('/search_images', methods=['GET', 'POST'])
def search_images(query_text=None):
    # Lấy truy vấn từ query string (GET) hoặc form (POST)
    if request.method == 'GET':
        query_text = request.args.get('query')
    elif request.method == 'POST':
        query_text = request.form.get('query')

    # Kiểm tra đầu vào
    if not query_text:
        # Trả về lỗi JSON, JavaScript sẽ hiển thị thông báo lỗi
        return jsonify({'error': 'Query text is required'}), 400

    # Truy vấn Qdrant để lấy kết quả
    qdrant_results = qdrant_manager.query_dataset(query_text)
    if not qdrant_results:
        # Trả về lỗi JSON nếu không tìm thấy kết quả
        return jsonify({'error': 'No images found. Try a different query.'}), 404

    # Xử lý kết quả, giải mã base64, và tạo dữ liệu cho JavaScript
    scenes = process_qdrant_results(qdrant_results)

    # Chuẩn bị dữ liệu JSON theo định dạng JavaScript mong đợi
    metadata_list = [scene['metadata'] for scene in scenes.values()]
    frame_paths = [scene['metadata']['frame_path'] for scene in scenes.values()]

    # Trả về JSON chứa frame_paths và metadata_list
    return jsonify({
        'frame_paths': frame_paths,  # Danh sách đường dẫn ảnh cho <img src>
        'metadata_list': metadata_list,  # Danh sách metadata cho nút Info
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)



# from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
# from vector_database import VectorDB
# import cv2
# import os
# from PIL import Image
# import base64
# from io import BytesIO
# import torch
# from torchvision import transforms
# import shutil
# from dotenv import load_dotenv

# app = Flask(__name__, static_folder='static', static_url_path='/static')

# load_dotenv()
# # Lấy API key từ biến môi trường
# QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
# if not QDRANT_API_KEY:
#     print("Cảnh báo: QDRANT_API_KEY không được cung cấp trong biến môi trường. Sử dụng giá trị mặc định.")

# qdrant_manager = VectorDB(
#     api='http://aienthusiasm:6333',
#     timeout=200.0,
#     api_key=QDRANT_API_KEY
# )

# # Định nghĩa hàm clear_directory trước khi sử dụng
# def clear_directory(directory):
#     for filename in os.listdir(directory):
#         file_path = os.path.join(directory, filename)
#         try:
#             if os.path.isfile(file_path) or os.path.islink(file_path):
#                 os.unlink(file_path)
#             elif os.path.isdir(file_path):
#                 shutil.rmtree(file_path)
#         except Exception as e:
#             print(f'Failed to delete {file_path}. Reason: {e}')

# image_folder = os.path.join('static', 'temporary', 'images')
# os.makedirs(image_folder, exist_ok=True)

# # Xóa các file hình ảnh tạm thời khi khởi động
# if os.path.exists(image_folder):
#     clear_directory(image_folder)

# @app.route('/')
# def home():
#     return render_template('newhome.html')

# @app.route('/static/<path:filename>')
# def custom_static(filename):
#     return send_from_directory('static', filename)

# @app.route('/search', methods=['POST'])
# def search():
#     query_text = request.form.get('query')

#     if not query_text:
#         return jsonify({'error': 'No query or files provided'}), 400

#     return search_images(query_text)

# def process_qdrant_results(qdrant_results):
#     scenes = {}
#     base64_list = []
#     output_paths = []
#     scene_data = []

#     # Chuẩn bị dữ liệu cho xử lý song song
#     for result in qdrant_results:
#         video_folder = result.payload['video_folder']
#         frame_number = result.payload['frame_number']
#         frame_idx = result.payload['frame_idx']
#         pts_time = result.payload['pts_time']
#         compressed = result.payload['compressed']

#         # Tạo đường dẫn
#         output_path = os.path.join('temporary', 'images', f"{video_folder}_{frame_number}.jpg").replace('\\', '/')
#         full_path = os.path.join('static', output_path)
#         relative_path = f"/static/{output_path}"

#         # Thêm vào danh sách xử lý
#         base64_list.append(compressed)
#         output_paths.append(full_path)
#         scene_data.append({
#             'video_folder': video_folder,
#             'frame_number': frame_number,
#             'frame_idx': frame_idx,
#             'pts_time': pts_time,
#             'relative_path': relative_path
#         })

#     # Xử lý song song việc giải mã base64
#     results = qdrant_manager.decode_images_parallel(base64_list, output_paths)

#     # Tạo scenes dict từ kết quả
#     for i, result in enumerate(results):
#         if result.startswith("✅"):  # Chỉ thêm vào scenes nếu xử lý thành công
#             data = scene_data[i]
#             scene_identifier = (data['video_folder'], data['frame_number'])
#             if scene_identifier not in scenes:
#                 scenes[scene_identifier] = {
#                     'metadata': {
#                         'video_folder': data['video_folder'],
#                         'frame_number': data['frame_number'],
#                         'frame_idx': data['frame_idx'],
#                         'pts_time': data['pts_time'],
#                         'frame_path': data['relative_path']
#                     }
#                 }

#     return scenes

# @app.route('/search_images', methods=['GET', 'POST'])
# def search_images(query_text=None):
#     if request.method == 'GET':
#         query_text = request.args.get('query')
#     elif request.method == 'POST':
#         query_text = request.form.get('query')

#     if not query_text:
#         return jsonify({'error': 'Query text are required'}), 400

#     qdrant_results = qdrant_manager.query_dataset(query_text)
#     if not qdrant_results:
#         return jsonify({'error': 'No images found. Try a different query.'}), 404

#     scenes = process_qdrant_results(qdrant_results)

#     metadata_list = [scene['metadata'] for scene in scenes.values()]
#     frame_paths = [scene['metadata']['frame_path'] for scene in scenes.values()]

#     return jsonify({
#         'frame_paths': frame_paths,
#         'metadata_list': metadata_list,
#     })

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5000)