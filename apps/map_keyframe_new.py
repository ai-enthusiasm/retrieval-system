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
BASE_DIR = Path(__file__).resolve().parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Đang sử dụng thiết bị: {device}")

# Đường dẫn dữ liệu
PATHS = {
    'csv_folder': BASE_DIR / 'data' / 'map-keyframes',
    'image_base_folder': BASE_DIR / 'data' / 'keyframe-image',
    'checkpoint_file': BASE_DIR / 'data' / 'checkpoint.json',
    'output_json_dir': BASE_DIR / 'data' / 'keyframes_json_test'
}

# Thiết lập tham số
MAX_BATCH_SIZE = 50
NEW_SIZE = (640, 360)

# Biến toàn cục để lưu trạng thái
CURRENT_STATE = {
    'subfolder': None,
    'output_json_path': None
}

# Đọc hoặc tạo checkpoint
def load_checkpoint():
    checkpoint_path = PATHS['checkpoint_file']
    if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
        logger.info(f"Checkpoint file '{checkpoint_path}' không tồn tại hoặc rỗng, tạo mới.")
        with checkpoint_path.open('w') as f:
            json.dump({}, f)
        return {}
    with checkpoint_path.open('r') as f:
        return json.load(f)

processed_subfolders = load_checkpoint()

# Khởi tạo mô hình ALIGN
processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
model = AlignModel.from_pretrained("kakaobrain/align-base").to(device)
model.eval()  # Đặt mô hình ở chế độ đánh giá để tối ưu hóa

# Hàm trích xuất đặc trưng từ ảnh với autocast để tối ưu GPU
def extract_features(images):
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad(), autocast():
        features = model.get_image_features(**inputs)
    return features.cpu().numpy().tolist()

# Hàm resize ảnh
def resize_image(image, size):
    return image.resize(size, Image.Resampling.LANCZOS)

# Hàm xử lý một ảnh
def process_image(img_path, new_size):
    try:
        image = Image.open(img_path).convert('RGB')
        resized_image = resize_image(image, new_size)
        return image, resized_image
    except Exception as e:
        logger.error(f"Lỗi khi xử lý ảnh {img_path}: {str(e)}")
        return None, None

# Hàm xử lý ảnh song song bằng ProcessPoolExecutor
def process_images_parallel(image_paths, new_size):
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(partial(process_image, new_size=new_size), image_paths))
    return results

# Hàm kiểm tra cấu trúc JSON
def validate_json_structure(data):
    required_fields = {'keyframes_folder', 'video_folder', 'frame_number', 'image_base64', 'image_filename', 'csv_data', 'vector'}
    csv_fields = {'pts_time', 'fps', 'frame_idx'}
    if not all(field in data for field in required_fields):
        logger.error(f"Thiếu trường trong JSON: {set(required_fields) - set(data.keys())}")
        return False
    if not all(field in data['csv_data'] for field in csv_fields):
        logger.error(f"Thiếu trường trong csv_data: {set(csv_fields) - set(data['csv_data'].keys())}")
        return False
    return True

# Hàm tạo thư mục
def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path

# Hàm tìm file CSV
def find_csv(sub_folder):
    csv_files = [f for f in PATHS['csv_folder'].glob(f"{sub_folder}*.csv")]
    if csv_files:
        logger.info(f"Tìm thấy file CSV: {csv_files[0]}")
        return csv_files[0]
    logger.warning(f"Không tìm thấy file CSV cho {sub_folder}")
    return None

# Hàm kiểm tra số lượng ảnh
def check_image_count(folder_path, csv_path):
    image_files = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.png'))
    df = pd.read_csv(csv_path)
    if len(image_files) != len(df):
        logger.error(f"Số lượng ảnh ({len(image_files)}) không khớp với CSV ({len(df)})")
        return False
    return True

# Hàm đếm tổng số điểm
def count_total_points():
    total = 0
    for folder in PATHS['image_base_folder'].iterdir():
        if folder.is_dir() and folder.name.startswith("Keyframes_L"):
            subfolder = folder / "keyframes"
            if subfolder.exists():
                for lxx_vxxx in subfolder.iterdir():
                    if lxx_vxxx.is_dir():
                        total += len(list(lxx_vxxx.glob('*.jpg'))) + len(list(lxx_vxxx.glob('*.png')))
    return total

# Hàm lưu dữ liệu vào JSON
def append_to_json(json_path, data_batch):
    with json_path.open('a') as f:
        for data in data_batch:
            json.dump(data, f, indent=4)
            f.write('\n')

# Hàm lưu checkpoint
def save_checkpoint():
    with PATHS['checkpoint_file'].open('w') as f:
        json.dump(processed_subfolders, f)
    logger.info("Đã lưu checkpoint")

# Xử lý Ctrl+C
def signal_handler(sig, frame):
    logger.info("Dừng chương trình...")
    if CURRENT_STATE['output_json_path'] and CURRENT_STATE['output_json_path'].exists():
        CURRENT_STATE['output_json_path'].unlink()
        logger.info(f"Đã xóa file JSON chưa hoàn thành: {CURRENT_STATE['output_json_path']}")
    save_checkpoint()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Main processing
def main():
    ensure_dir(PATHS['output_json_dir'])
    total_points = count_total_points()
    processed_points = 0

    for keyframes_folder in PATHS['image_base_folder'].iterdir():
        if not (keyframes_folder.is_dir() and keyframes_folder.name.startswith("Keyframes_L")):
            continue

        logger.info(f"Đang xử lý thư mục chính: {keyframes_folder.name}")
        subfolder_path = keyframes_folder / "keyframes"
        if not subfolder_path.exists():
            continue

        json_output_dir = ensure_dir(PATHS['output_json_dir'] / keyframes_folder.name)
        processed_subs = processed_subfolders.get(keyframes_folder.name, [])

        for lxx_vxxx in subfolder_path.iterdir():
            if not lxx_vxxx.is_dir() or lxx_vxxx.name in processed_subs:
                if lxx_vxxx.name in processed_subs:
                    logger.info(f"Thư mục {lxx_vxxx.name} đã xử lý, bỏ qua.")
                continue

            logger.info(f"Đang xử lý thư mục con: {lxx_vxxx.name}")
            CURRENT_STATE['subfolder'] = lxx_vxxx.name

            csv_path = find_csv(lxx_vxxx.name)
            if not csv_path:
                continue

            if not check_image_count(lxx_vxxx, csv_path):
                raise SystemExit("Dừng do số lượng ảnh không khớp.")

            df = pd.read_csv(csv_path)
            json_path = json_output_dir / f"{lxx_vxxx.name}.json"
            CURRENT_STATE['output_json_path'] = json_path

            if json_path.exists():
                json_path.unlink()
                logger.info(f"Xóa file JSON cũ: {json_path}")

            json_path.touch()

            current_batch = []
            for _, row in df.iterrows():
                frame_number = int(row['n'])
                img_path = lxx_vxxx / f"{frame_number:03d}.jpg"

                if not img_path.exists():
                    logger.warning(f"Ảnh không tồn tại: {img_path}")
                    continue

                current_batch.append((img_path, frame_number, row))

                if len(current_batch) >= MAX_BATCH_SIZE:
                    process_batch(current_batch, json_path, keyframes_folder.name, lxx_vxxx.name)
                    current_batch = []

            if current_batch:
                process_batch(current_batch, json_path, keyframes_folder.name, lxx_vxxx.name)

            processed_points += len(df)
            logger.info(f"Đã xử lý {processed_points}/{total_points} points.")

            processed_subs.append(lxx_vxxx.name)
            processed_subfolders[keyframes_folder.name] = processed_subs
            save_checkpoint()
            CURRENT_STATE.update(subfolder=None, output_json_path=None)

    logger.info("Hoàn tất xử lý.")

# Xử lý batch
def process_batch(batch, json_path, keyframes_folder, video_folder):
    logger.info(f"Xử lý batch với {len(batch)} ảnh...")
    image_paths = [item[0] for item in batch]

    # Xử lý song song
    processed_images = process_images_parallel(image_paths, NEW_SIZE)
    original_images = [img[0] for img in processed_images if img[0] is not None]
    resized_images = [img[1] for img in processed_images if img[1] is not None]

    # Trích xuất đặc trưng
    features = extract_features(original_images) if original_images else []

    data_batch = []
    feature_idx = 0
    for (img_path, frame_number, row), (orig_img, resized_img) in zip(batch, processed_images):
        if orig_img is None or resized_img is None:
            logger.warning(f"Bỏ qua ảnh lỗi: {img_path}")
            continue

        with img_path.open('rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')

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
            "vector": features[feature_idx]
        }

        if not validate_json_structure(frame_info):
            logger.error(f"JSON không hợp lệ tại frame {frame_number}")
            if json_path.exists():
                json_path.unlink()
            save_checkpoint()
            raise SystemExit("Dừng do JSON không hợp lệ.")

        data_batch.append(frame_info)
        feature_idx += 1

    append_to_json(json_path, data_batch)
    logger.info("Batch xử lý xong.")

if __name__ == "__main__":
    main()