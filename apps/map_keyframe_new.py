import os
import base64
import json
import pandas as pd
import torch
import logging
from PIL import Image
from transformers import AlignProcessor, AlignModel
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import signal
import sys
from torch.cuda.amp import autocast
from pathlib import Path
import numpy as np

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Định nghĩa thư mục và thiết bị
BASE_DIR = Path(__file__).resolve().parent.parent  # Lấy thư mục cha của thư mục chứa file Python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Chọn GPU nếu có, nếu không dùng CPU
logger.info(f"Đang sử dụng thiết bị: {device}")  # Ghi log về thiết bị được sử dụng

# Định nghĩa các đường dẫn đến thư mục và file
PATHS = {
    'csv_folder': BASE_DIR / 'data' / 'map-keyframes',  # Thư mục chứa file CSV
    'image_base_folder': BASE_DIR / 'data' / 'keyframe-image',  # Thư mục chứa ảnh keyframes
    'checkpoint_file': BASE_DIR / 'data' / 'checkpoint.json',  # File lưu trạng thái xử lý
    'output_json_dir': BASE_DIR / 'data' / 'keyframes_json_test'  # Thư mục lưu file JSON đầu ra
}

# Thiết lập các tham số cố định
MAX_BATCH_SIZE = 50  # Số lượng ảnh tối đa trong một batch khi xử lý
NEW_SIZE = (640, 360)  # Kích thước mới của ảnh sau khi resize

# Biến toàn cục để theo dõi trạng thái hiện tại
CURRENT_STATE = {
    'subfolder': None,  # Tên thư mục con đang xử lý
    'output_json_path': None  # Đường dẫn file JSON đầu ra đang ghi
}

# Hàm đọc hoặc tạo file checkpoint
def load_checkpoint():
    checkpoint_path = PATHS['checkpoint_file']
    if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
        logger.info(f"Checkpoint file '{checkpoint_path}' không tồn tại hoặc rỗng, tạo mới.")
        with checkpoint_path.open('w') as f:
            json.dump({}, f)  # Tạo file JSON rỗng
        return {}
    with checkpoint_path.open('r') as f:
        return json.load(f)  # Đọc nội dung checkpoint

processed_subfolders = load_checkpoint()  # Tải danh sách các thư mục đã xử lý

# Khởi tạo mô hình và processor ALIGN từ Hugging Face
processor = AlignProcessor.from_pretrained("kakaobrain/align-base")  # Processor để xử lý đầu vào (ảnh)
model = AlignModel.from_pretrained("kakaobrain/align-base").to(device)  # Mô hình ALIGN để trích xuất đặc trưng
model.eval()  # Chuyển mô hình sang chế độ đánh giá (tối ưu cho inference)

# Hàm trích xuất đặc trưng từ danh sách ảnh
def extract_features(images):
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)  # Chuẩn bị dữ liệu ảnh
    with torch.no_grad(), autocast():  # Tắt gradient và dùng autocast để tối ưu GPU
        features = model.get_image_features(**inputs)  # Trích xuất đặc trưng
    return features.cpu().numpy().tolist()  # Chuyển đặc trưng về dạng list

# Hàm resize ảnh
def resize_image(image, size):
    return image.resize(size, Image.Resampling.LANCZOS)  # Resize với phương pháp LANCZOS để giữ chất lượng
    
# Hàm xử lý một ảnh (mở và resize)
def process_image(img_path, new_size):
    try:
        image = Image.open(img_path).convert('RGB')  # Mở ảnh và chuyển sang RGB
        resized_image = resize_image(image, new_size)  # Resize ảnh
        return image, resized_image  # Trả về ảnh gốc và ảnh đã resize
    except Exception as e:
        logger.error(f"Lỗi khi xử lý ảnh {img_path}: {str(e)}")  # Ghi log nếu có lỗi
        return None, None  # Trả về None nếu lỗi

# Hàm xử lý nhiều ảnh song song bằng ProcessPoolExecutor
def process_images_parallel(image_paths, new_size):
    with ProcessPoolExecutor() as executor:  # Tạo pool xử lý song song
        results = list(executor.map(partial(process_image, new_size=new_size), image_paths))  # Áp dụng process_image cho từng ảnh
    return results  # Trả về danh sách kết quả [(image, resized_image), ...]

# Hàm kiểm tra xem JSON có đầy đủ các trường cần thiết không
def validate_json_structure(data):
    required_fields = {'keyframes_folder', 'video_folder', 'frame_number', 'image_base64', 'image_filename', 'csv_data', 'vector'}  # Các trường bắt buộc trong JSON
    csv_fields = {'pts_time', 'fps', 'frame_idx'}  # Các trường bắt buộc trong csv_data
    if not all(field in data for field in required_fields):
        logger.error(f"Thiếu trường trong JSON: {set(required_fields) - set(data.keys())}")  # Ghi log nếu thiếu trường
        return False
    if not all(field in data['csv_data'] for field in csv_fields):
        logger.error(f"Thiếu trường trong csv_data: {set(csv_fields) - set(data['csv_data'].keys())}")  # Ghi log nếu thiếu trường trong csv_data
        return False
    return True  # Trả về True nếu JSON hợp lệ

# Hàm tạo thư mục nếu chưa tồn tại
def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)  # Tạo thư mục và các thư mục cha nếu cần
    return path

# Hàm tìm file CSV tương ứng với thư mục con
def find_csv(sub_folder):
    csv_files = [f for f in PATHS['csv_folder'].glob(f"{sub_folder}*.csv")]  # Tìm tất cả file CSV bắt đầu bằng sub_folder
    if csv_files:
        logger.info(f"Tìm thấy file CSV: {csv_files[0]}")  # Ghi log file tìm thấy
        return csv_files[0]
    logger.warning(f"Không tìm thấy file CSV cho {sub_folder}")  # Ghi log nếu không tìm thấy
    return None

# Hàm kiểm tra số lượng ảnh có khớp với số dòng trong CSV không
def check_image_count(folder_path, csv_path):
    image_files = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.png'))  # Lấy danh sách file ảnh (jpg, png)
    df = pd.read_csv(csv_path)  # Đọc file CSV
    if len(image_files) != len(df):
        logger.error(f"Số lượng ảnh ({len(image_files)}) không khớp với CSV ({len(df)})")  # Ghi log nếu không khớp
        return False
    return True  # Trả về True nếu số lượng khớp

# Hàm đếm tổng số ảnh trong tất cả thư mục con
def count_total_points():
    total = 0
    for folder in PATHS['image_base_folder'].iterdir():
        if folder.is_dir() and folder.name.startswith("Keyframes_L"):  # Chỉ xử lý thư mục bắt đầu bằng "Keyframes_L"
            subfolder = folder / "keyframes"
            if subfolder.exists():
                for lxx_vxxx in subfolder.iterdir():
                    if lxx_vxxx.is_dir():
                        total += len(list(lxx_vxxx.glob('*.jpg'))) + len(list(lxx_vxxx.glob('*.png')))  # Đếm ảnh jpg và png
    return total  # Trả về tổng số ảnh

# Hàm thêm dữ liệu vào file JSON
def append_to_json(json_path, data_batch):
    with json_path.open('a') as f:  # Mở file ở chế độ append
        for data in data_batch:
            json.dump(data, f, indent=4)  # Ghi từng đối tượng JSON
            f.write('\n')  # Thêm dòng mới để dễ đọc

# Hàm lưu trạng thái xử lý vào checkpoint
def save_checkpoint():
    with PATHS['checkpoint_file'].open('w') as f:
        json.dump(processed_subfolders, f)  # Ghi danh sách thư mục đã xử lý
    logger.info("Đã lưu checkpoint")  # Ghi log

# Hàm xử lý tín hiệu ngắt (Ctrl+C)
def signal_handler(sig, frame):
    logger.info("Dừng chương trình...")  # Ghi log khi chương trình bị dừng
    if CURRENT_STATE['output_json_path'] and CURRENT_STATE['output_json_path'].exists():
        CURRENT_STATE['output_json_path'].unlink()  # Xóa file JSON chưa hoàn thành
        logger.info(f"Đã xóa file JSON chưa hoàn thành: {CURRENT_STATE['output_json_path']}")
    save_checkpoint()  # Lưu checkpoint trước khi thoát
    sys.exit(0)  # Thoát chương trình

signal.signal(signal.SIGINT, signal_handler)  # Gán hàm xử lý cho tín hiệu SIGINT


# Hàm xử lý một batch ảnh
def process_batch(batch, json_path, keyframes_folder, video_folder):
    logger.info(f"Xử lý batch với {len(batch)} ảnh...")
    image_paths = [item[0] for item in batch]  # Lấy danh sách đường dẫn ảnh

    # Xử lý song song các ảnh
    processed_images = process_images_parallel(image_paths, NEW_SIZE)
    original_images = [img[0] for img in processed_images if img[0] is not None]  # Ảnh gốc
    resized_images = [img[1] for img in processed_images if img[1] is not None]  # Ảnh đã resize

    # Trích xuất đặc trưng từ ảnh gốc
    features = extract_features(original_images) if original_images else []

    data_batch = []  # Danh sách dữ liệu JSON
    feature_idx = 0
    for (img_path, frame_number, row), (orig_img, resized_img) in zip(batch, processed_images):
        if orig_img is None or resized_img is None:
            logger.warning(f"Bỏ qua ảnh lỗi: {img_path}")  # Bỏ qua ảnh lỗi
            continue

        with img_path.open('rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')  # Mã hóa ảnh thành base64

        # Tạo đối tượng JSON cho frame
        frame_info = {
            "keyframes_folder": keyframes_folder,
            "video_folder": video_folder,
            "frame_number": frame_number,
            "image_base64": image_base64,
            "image_filename": img_path.name,
            "csv_data": {
                "pts_time": row['pts_time'],
                "fps": row['fps'],
                "frame_idx": row['frame_idx']
            },
            "vector": features[feature_idx]  # Đặc trưng của ảnh
        }

        if not validate_json_structure(frame_info):
            logger.error(f"JSON không hợp lệ tại frame {frame_number}")
            if json_path.exists():
                json_path.unlink()  # Xóa file JSON nếu JSON không hợp lệ
            save_checkpoint()  # Lưu checkpoint
            raise SystemExit("Dừng do JSON không hợp lệ.")  # Thoát chương trình

        data_batch.append(frame_info)  # Thêm vào batch
        feature_idx += 1

    append_to_json(json_path, data_batch)  # Lưu batch vào file JSON
    logger.info("Batch xử lý xong.")

# Hàm chính điều khiển toàn bộ quy trình
def main():
    ensure_dir(PATHS['output_json_dir'])  # Tạo thư mục đầu ra nếu chưa có
    total_points = count_total_points()  # Đếm tổng số ảnh cần xử lý
    processed_points = 0  # Biến đếm số ảnh đã xử lý

    # Duyệt qua các thư mục chính (Keyframes_Lxx)
    for keyframes_folder in PATHS['image_base_folder'].iterdir():
        if not (keyframes_folder.is_dir() and keyframes_folder.name.startswith("Keyframes_L")):
            continue  # Bỏ qua nếu không phải thư mục hoặc không bắt đầu bằng "Keyframes_L"

        logger.info(f"Đang xử lý thư mục chính: {keyframes_folder.name}")
        subfolder_path = keyframes_folder / "keyframes"  # Thư mục con "keyframes"
        if not subfolder_path.exists():
            continue  # Bỏ qua nếu không có thư mục "keyframes"

        json_output_dir = ensure_dir(PATHS['output_json_dir'] / keyframes_folder.name)  # Tạo thư mục đầu ra cho thư mục chính
        processed_subs = processed_subfolders.get(keyframes_folder.name, [])  # Lấy danh sách thư mục con đã xử lý

        # Duyệt qua các thư mục con (Lxx_Vxxx)
        for lxx_vxxx in subfolder_path.iterdir():
            if not lxx_vxxx.is_dir() or lxx_vxxx.name in processed_subs:
                if lxx_vxxx.name in processed_subs:
                    logger.info(f"Thư mục {lxx_vxxx.name} đã xử lý, bỏ qua.")  # Bỏ qua nếu đã xử lý
                continue

            logger.info(f"Đang xử lý thư mục con: {lxx_vxxx.name}")
            CURRENT_STATE['subfolder'] = lxx_vxxx.name  # Cập nhật trạng thái

            csv_path = find_csv(lxx_vxxx.name)  # Tìm file CSV
            if not csv_path:
                continue  # Bỏ qua nếu không tìm thấy CSV

            if not check_image_count(lxx_vxxx, csv_path):
                raise SystemExit("Dừng do số lượng ảnh không khớp.")  # Thoát nếu số lượng ảnh không khớp

            df = pd.read_csv(csv_path)  # Đọc file CSV
            json_path = json_output_dir / f"{lxx_vxxx.name}.json"  # Đường dẫn file JSON đầu ra
            CURRENT_STATE['output_json_path'] = json_path

            if json_path.exists():
                json_path.unlink()  # Xóa file JSON cũ nếu có
                logger.info(f"Xóa file JSON cũ: {json_path}")

            json_path.touch()  # Tạo file JSON mới

            current_batch = []  # Danh sách batch hiện tại
            for _, row in df.iterrows():
                frame_number = int(row['n'])  # Lấy số frame từ cột 'n'
                img_path = lxx_vxxx / f"{frame_number:03d}.jpg"  # Đường dẫn ảnh (định dạng 001.jpg, 002.jpg, ...)

                if not img_path.exists():
                    logger.warning(f"Ảnh không tồn tại: {img_path}")  # Ghi log nếu ảnh không tồn tại
                    continue

                current_batch.append((img_path, frame_number, row))  # Thêm vào batch

                if len(current_batch) >= MAX_BATCH_SIZE:
                    process_batch(current_batch, json_path, keyframes_folder.name, lxx_vxxx.name)  # Xử lý batch
                    current_batch = []  # Reset batch

            if current_batch:
                process_batch(current_batch, json_path, keyframes_folder.name, lxx_vxxx.name)  # Xử lý batch còn lại

            processed_points += len(df)  # Cập nhật số điểm đã xử lý
            logger.info(f"Đã xử lý {processed_points}/{total_points} points.")  # Ghi log tiến độ

            processed_subs.append(lxx_vxxx.name)  # Thêm thư mục con vào danh sách đã xử lý
            processed_subfolders[keyframes_folder.name] = processed_subs
            save_checkpoint()  # Lưu checkpoint
            CURRENT_STATE.update(subfolder=None, output_json_path=None)  # Reset trạng thái

    logger.info("Hoàn tất xử lý.")  # Ghi log khi hoàn tất

if __name__ == "__main__":
    main()