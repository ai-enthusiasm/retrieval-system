import os
import json
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import re
import pickle
from retry import retry

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('import.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Định nghĩa thư mục và cấu hình
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent
KEYFRAMES_FOLDER = BASE_DIR / 'data' / 'keyframes_json_test'
BATCH_SIZE = 50
COLLECTION_NAME = "dataset"
VECTOR_SIZE = 640

# Đường dẫn file lưu UUID
UUID_CACHE_FILE = BASE_DIR / 'data' / 'uuid_cache.pkl'

# Load hoặc tạo danh sách UUID
def load_uuid_cache():
    if UUID_CACHE_FILE.exists():
        with UUID_CACHE_FILE.open('rb') as f:
            return pickle.load(f)
    return set()

used_uuids = load_uuid_cache()

# Lưu UUID cache
def save_uuid_cache():
    with UUID_CACHE_FILE.open('wb') as f:
        pickle.dump(used_uuids, f)
    logger.info("Đã lưu UUID cache")

# Lấy API key từ biến môi trường
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
if not QDRANT_API_KEY:
    logger.warning("QDRANT_API_KEY không được cung cấp, sử dụng giá trị mặc định.")

# Kết nối đến Qdrant
client = QdrantClient(
    host="localhost",
    port=6333,
    api_key=QDRANT_API_KEY,
    https=False
)

# Tạo collection nếu chưa tồn tại
def init_collection():
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        logger.info(f"Tạo collection {COLLECTION_NAME}")

# Tạo UUID duy nhất
def generate_unique_uuid():
    new_uuid = str(uuid.uuid4())
    while new_uuid in used_uuids:
        new_uuid = str(uuid.uuid4())
    used_uuids.add(new_uuid)
    return new_uuid

# Sửa lỗi JSON
def fix_json_format(file_path):
    try:
        with file_path.open('r', encoding='utf-8') as f:
            content = f.read().strip()

        if content.startswith("[") and content.endswith("]"):
            return file_path

        # Sử dụng regex để thay thế các object JSON liên tiếp
        fixed_json = re.sub(r'}\s*{', '},{', content)
        fixed_json = f"[{fixed_json}]"

        with file_path.open('w', encoding='utf-8') as f:
            f.write(fixed_json)

        logger.info(f"Đã sửa JSON: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Lỗi khi sửa JSON {file_path}: {e}")
        return None

# Chuyển đổi dữ liệu JSON sang PointStruct
def process_json_data(data, keyframes_folder):
    try:
        payload = {
            "keyframes_folder": keyframes_folder,
            "video_folder": data['video_folder'],
            "frame_number": data['frame_number'],
            "pts_time": data['csv_data']['pts_time'],
            "frame_idx": data['csv_data']['frame_idx'],
            "compressed": data['image_base64']
        }
        return models.PointStruct(
            id=generate_unique_uuid(),
            vector=data['vector'],
            payload=payload
        )
    except KeyError as e:
        logger.error(f"Dữ liệu JSON thiếu trường: {e}")
        return None

# Xử lý một file JSON
def process_json_file(file_path, keyframes_folder):
    points = []
    fixed_file_path = fix_json_format(file_path)
    if not fixed_file_path:
        logger.warning(f"Bỏ qua file lỗi: {file_path}")
        return points

    try:
        with fixed_file_path.open('r', encoding='utf-8') as f:
            json_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Lỗi đọc JSON {file_path}: {e}")
        return points

    for data in json_data:
        point = process_json_data(data, keyframes_folder)
        if point:
            points.append(point)

    return points

# Hàm upsert với retry
@retry(tries=3, delay=1, backoff=2)
def upsert_points(points):
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    logger.info(f"Upserted {len(points)} points")

# Xử lý song song các file JSON
def process_folder(keyframes_folder):
    folder_path = KEYFRAMES_FOLDER / keyframes_folder
    points = []
    total_imported = 0

    if not (folder_path.is_dir() and keyframes_folder.startswith("Keyframes_L")):
        return 0

    logger.info(f"Xử lý thư mục: {keyframes_folder}")
    json_files = [f for f in folder_path.glob('*.json')]

    with ProcessPoolExecutor() as executor:
        process_func = partial(process_json_file, keyframes_folder=keyframes_folder)
        results = executor.map(process_func, json_files)

    for batch_points in results:
        points.extend(batch_points)

        while len(points) >= BATCH_SIZE:
            batch = points[:BATCH_SIZE]
            try:
                upsert_points(batch)
                total_imported += len(batch)
                points = points[BATCH_SIZE:]
            except Exception as e:
                logger.error(f"Lỗi khi upsert batch: {e}")
                save_uuid_cache()
                raise

    if points:
        try:
            upsert_points(points)
            total_imported += len(points)
        except Exception as e:
            logger.error(f"Lỗi khi upsert batch cuối: {e}")
            save_uuid_cache()
            raise

    logger.info(f"Imported {total_imported} points từ {keyframes_folder}")
    return total_imported

# Main processing
def main():
    init_collection()
    total_imported = 0

    folders = [f for f in KEYFRAMES_FOLDER.iterdir() if f.is_dir() and f.name.startswith("Keyframes_L")]
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_folder, [f.name for f in folders])
        total_imported = sum(results)

    save_uuid_cache()
    logger.info(f"Tổng số điểm dữ liệu đã nhập: {total_imported}")

if __name__ == "__main__":
    main()