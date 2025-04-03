import os
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Cấu hình
KEYFRAMES_FOLDER = os.path.join(BASE_DIR, 'data', 'keyframes_json')
BATCH_SIZE = 50
COLLECTION_NAME = "dataset"
VECTOR_SIZE = 640
used_uuids = set()

# Lấy API key từ biến môi trường
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
if not QDRANT_API_KEY:
    print("Cảnh báo: QDRANT_API_KEY không được cung cấp trong biến môi trường. Sử dụng giá trị mặc định.")

# Kết nối đến Qdrant
from qdrant_client import QdrantClient

client = QdrantClient(
    host="localhost",
    port=6333,
    api_key=QDRANT_API_KEY,
    https=False
)

# Tạo collection nếu chưa tồn tại
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
    )

def generate_unique_uuid():
    """Tạo UUID duy nhất để tránh trùng lặp dữ liệu"""
    new_uuid = str(uuid.uuid4())
    while new_uuid in used_uuids:
        new_uuid = str(uuid.uuid4())
    used_uuids.add(new_uuid)
    return new_uuid

def fix_json_format(file_path):
    """Sửa lỗi JSON bằng cách thêm `[` `]` và thay dòng mới bằng dấu phẩy"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()

        # Nếu file JSON đã có `[` `]` thì không cần sửa
        if content.startswith("[") and content.endswith("]"):
            return file_path  # File JSON đúng format

        # Thêm `[` đầu file và `]` cuối file, thay `}\n{` thành `},\n{`
        fixed_json = "[" + content.replace("}\n{", "},\n{") + "]"

        # Ghi đè file JSON
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(fixed_json)

        print(f"✅ Đã sửa lỗi JSON: {file_path}")
        return file_path
    except Exception as e:
        print(f"⚠️ Lỗi khi sửa file {file_path}: {e}")
        return None

def process_json_data(data, keyframes_folder):
    """Chuyển đổi dữ liệu JSON thành format phù hợp cho Qdrant"""
    payload = {
        "keyframes_folder": keyframes_folder,  # Lưu thông tin thư mục cha
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

def import_data_in_batches():
    """Duyệt qua tất cả thư mục Keyframes_Lxx và import dữ liệu từ các file JSON"""
    points = []
    total_imported = 0

    # Duyệt qua từng thư mục Keyframes_Lxx
    for keyframes_folder in os.listdir(KEYFRAMES_FOLDER):
        keyframes_path = os.path.join(KEYFRAMES_FOLDER, keyframes_folder)

        # Kiểm tra nếu nó là một thư mục hợp lệ
        if os.path.isdir(keyframes_path) and keyframes_folder.startswith("Keyframes_L"):
            print(f"📂 Đang xử lý thư mục: {keyframes_folder}")

            # Duyệt qua từng file JSON bên trong thư mục Keyframes_Lxx
            for filename in os.listdir(keyframes_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(keyframes_path, filename)

                    # Sửa lỗi JSON trước khi đọc
                    fixed_file_path = fix_json_format(file_path)
                    if not fixed_file_path:
                        print(f"⚠️ Bỏ qua file lỗi: {filename}")
                        continue

                    # Đọc JSON sau khi fix
                    with open(fixed_file_path, 'r', encoding='utf-8') as f:
                        try:
                            json_data = json.load(f)  # Kiểm tra định dạng JSON
                        except json.JSONDecodeError as e:
                            print(f"⚠️ Lỗi khi đọc {filename}: {e}")
                            continue  # Bỏ qua file bị lỗi

                    # Xử lý từng object JSON trong danh sách
                    for data in json_data:
                        point = process_json_data(data, keyframes_folder)
                        points.append(point)

                        # Import theo batch
                        if len(points) >= BATCH_SIZE:
                            client.upsert(
                                collection_name=COLLECTION_NAME,
                                points=points
                            )
                            total_imported += len(points)
                            print(f"✅ Imported {total_imported} points")
                            points = []

    # Import các điểm còn lại
    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        total_imported += len(points)
        print(f"✅ Imported {total_imported} points")

    print(f"🎉 Tổng số điểm dữ liệu đã nhập: {total_imported}")

if __name__ == "__main__":
    import_data_in_batches()
