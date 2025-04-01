import os
import base64
import json
import pandas as pd
import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Đang sử dụng thiết bị: {device}")
# Đường dẫn dữ liệu
csv_folder = os.path.join(BASE_DIR, 'data', 'map-keyframes')
image_base_folder = os.path.join(BASE_DIR, 'data', 'keyframe-image')
checkpoint_file = os.path.join(BASE_DIR, 'data', 'checkpoint.json')
output_json_dir = os.path.join(BASE_DIR, 'data', 'keyframes_json')

# Giới hạn số lượng batch là 20 ảnh
max_batch_size_in_images = 50  # Batch chứa tối đa 20 ảnh

# Biến toàn cục để lưu tên thư mục con và đường dẫn file JSON đang xử lý
current_subfolder = None
current_output_json_path = None

# Kiểm tra và tạo file checkpoint nếu nó không tồn tại hoặc rỗng
if not os.path.exists(checkpoint_file) or os.path.getsize(checkpoint_file) == 0:
    print(f"Checkpoint file '{checkpoint_file}' không tồn tại hoặc rỗng, tạo file mới.")
    with open(checkpoint_file, 'w') as f:
        json.dump({}, f)
    processed_subfolders = {}
else:
    # Đọc checkpoint nếu có và không rỗng
    with open(checkpoint_file, 'r') as f:
        processed_subfolders = json.load(f)

# Kích thước mới cho ảnh sau khi resize để lưu vào JSON
new_size = (640, 360)

# Khởi tạo mô hình ALIGN
processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
model = AlignModel.from_pretrained("kakaobrain/align-base").to(device)

# Hàm trích xuất đặc trưng từ ảnh gốc 1280x720
def extract_features_from_original_images(images):
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features.cpu().numpy().tolist()

# Hàm resize ảnh nhưng không lưu lại vào thư mục, chỉ để chuyển sang Base64
def resize_image(image, size):
    return image.resize(size)

# Hàm xử lý ảnh song song bằng ThreadPoolExecutor
def process_images_in_parallel(current_batch, new_size):
    with ThreadPoolExecutor() as executor:
        processed_images = list(executor.map(lambda item: process_image(item[0], new_size), current_batch))
    return processed_images

# Hàm xử lý một ảnh (resize cho JSON và giữ nguyên ảnh gốc cho trích xuất đặc trưng)
def process_image(img_path, new_size):
    try:
        image = Image.open(img_path).convert('RGB')  # Đọc ảnh gốc 1280x720
        resized_image = resize_image(image, new_size)  # Resize để lưu vào JSON
        return resized_image
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {img_path}: {str(e)}")
        return None

# Hàm kiểm tra cấu trúc JSON
def validate_json_structure(data):
    required_fields = ['video_folder', 'frame_number', 'csv_data', 'image_base64', 'vector']
    csv_required_fields = ['pts_time', 'frame_idx']
    
    # Kiểm tra các trường chính
    for field in required_fields:
        if field not in data:
            print(f"Lỗi: Thiếu trường '{field}' trong JSON data.")
            return False
    
    # Kiểm tra các trường trong 'csv_data'
    for csv_field in csv_required_fields:
        if csv_field not in data['csv_data']:
            print(f"Lỗi: Thiếu trường '{csv_field}' trong 'csv_data'.")
            return False
    
    # Nếu không có lỗi
    return True

# Đảm bảo thư mục JSON tồn tại
if not os.path.exists(output_json_dir):
    os.makedirs(output_json_dir)

# Hàm tạo thư mục con cho từng thư mục lớn keyframe_Lxx
def create_subfolder_if_not_exists(keyframes_folder):
    subfolder_path = os.path.join(output_json_dir, keyframes_folder)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    return subfolder_path

# Hàm tìm file CSV dựa vào tên thư mục Lxx_Vxxx
def find_csv_for_keyframe(sub_folder):
    for csv_filename in os.listdir(csv_folder):
        if csv_filename.startswith(sub_folder) and csv_filename.endswith('.csv'):
            csv_file_path = os.path.join(csv_folder, csv_filename)
            if os.path.exists(csv_file_path):
                print(f"Tìm thấy file CSV '{csv_filename}' khớp với thư mục '{sub_folder}'")
                return csv_file_path
    print(f"Không tìm thấy file CSV khớp với thư mục '{sub_folder}'")
    return None

# Hàm kiểm tra số lượng ảnh trong folder và trong CSV
def check_image_count(folder_path, csv_file_path):
    # Đếm số lượng ảnh trong folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
    image_count = len(image_files)

    # Đếm số lượng ảnh trong file CSV
    df = pd.read_csv(csv_file_path)
    csv_image_count = len(df)

    # So sánh số lượng ảnh
    if image_count != csv_image_count:
        print(f"Lỗi: Số lượng ảnh trong folder ({image_count}) và trong CSV ({csv_image_count}) không khớp!")
        return False
    return True

# Hàm đếm tổng số point của tất cả các folder
def count_total_points(image_base_folder):
    total_points = 0
    for keyframes_folder in os.listdir(image_base_folder):
        keyframes_folder_path = os.path.join(image_base_folder, keyframes_folder)
        if os.path.isdir(keyframes_folder_path) and keyframes_folder.startswith("Keyframes_L"):
            keyframes_subfolder_path = os.path.join(keyframes_folder_path, "keyframes")
            if os.path.exists(keyframes_subfolder_path):
                for lxx_vxxx_folder in os.listdir(keyframes_subfolder_path):
                    lxx_vxxx_path = os.path.join(keyframes_subfolder_path, lxx_vxxx_folder)
                    if os.path.isdir(lxx_vxxx_path):
                        image_files = [f for f in os.listdir(lxx_vxxx_path) if f.endswith(('.jpg', '.png'))]
                        total_points += len(image_files)
    return total_points

# Hàm lưu dữ liệu khi xử lý xong từng batch vào JSON
def append_to_video_json(output_json_path, data_batch):
    with open(output_json_path, 'a') as json_file:
        for data in data_batch:
            json.dump(data, json_file, indent=4)  # Ghi mỗi đối tượng JSON đẹp mắt với indent
            json_file.write("\n")  # Ghi từng đối tượng JSON trên mỗi dòng để tránh lặp

# Hàm lưu checkpoint khi dừng chương trình
def save_checkpoint():
    with open(checkpoint_file, 'w') as f:
        json.dump(processed_subfolders, f)
    print("Đã lưu checkpoint.")

# Hàm xử lý khi nhấn Ctrl+C
def signal_handler(sig, frame):
    print("Dừng chương trình...")

    # Xóa file JSON của thư mục con đang xử lý nếu có
    if current_output_json_path and os.path.exists(current_output_json_path):
        os.remove(current_output_json_path)
        print(f"Đã xóa file JSON chưa hoàn thành: {current_output_json_path}")

    save_checkpoint()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Lặp qua tất cả các thư mục Keyframes_Lxx trong image_base_folder
total_points = count_total_points(image_base_folder)
processed_points = 0

for keyframes_folder in os.listdir(image_base_folder):
    keyframes_folder_path = os.path.join(image_base_folder, keyframes_folder)
    
    if os.path.isdir(keyframes_folder_path) and keyframes_folder.startswith("Keyframes_L"):
        print(f"Đang xử lý thư mục chính: {keyframes_folder}")
        
        keyframes_subfolder_path = os.path.join(keyframes_folder_path, "keyframes")
        if os.path.exists(keyframes_subfolder_path):
            # Tạo thư mục con cho keyframes_folder trong output_json_dir
            subfolder_output_json_dir = create_subfolder_if_not_exists(keyframes_folder)
            
            # Lấy danh sách các thư mục con đã xử lý trong keyframes_folder từ checkpoint
            processed_subs_in_folder = processed_subfolders.get(keyframes_folder, [])

            for lxx_vxxx_folder in os.listdir(keyframes_subfolder_path):
                lxx_vxxx_path = os.path.join(keyframes_subfolder_path, lxx_vxxx_folder)
                
                if os.path.isdir(lxx_vxxx_path):
                    # Kiểm tra nếu thư mục con này đã được xử lý
                    if lxx_vxxx_folder in processed_subs_in_folder:
                        print(f"Thư mục con {lxx_vxxx_folder} đã được xử lý, bỏ qua.")
                        continue

                    print(f"Đang xử lý thư mục con: {lxx_vxxx_folder}")

                    # Cập nhật biến toàn cục
                    current_subfolder = lxx_vxxx_folder

                    csv_file_path = find_csv_for_keyframe(lxx_vxxx_folder)
                    if csv_file_path:
                        # Kiểm tra số lượng ảnh trong folder và file CSV
                        if not check_image_count(lxx_vxxx_path, csv_file_path):
                            print(f"Dừng chương trình do số lượng ảnh không khớp cho thư mục {lxx_vxxx_folder}")
                            raise SystemExit("Dừng chương trình do phát hiện lỗi số lượng ảnh không khớp.")

                        df = pd.read_csv(csv_file_path)

                        # Tạo sẵn file JSON cho thư mục con trong thư mục keyframes_folder
                        output_json_path = os.path.join(subfolder_output_json_dir, f'{lxx_vxxx_folder}.json')
                        current_output_json_path = output_json_path  # Cập nhật biến toàn cục

                        # Kiểm tra nếu file JSON đã tồn tại nhưng thư mục con chưa được đánh dấu là đã xử lý
                        if os.path.exists(output_json_path):
                            print(f"File JSON {output_json_path} đã tồn tại nhưng chưa được đánh dấu là đã xử lý. Xóa file và xử lý lại.")
                            os.remove(output_json_path)

                        # Tạo file JSON mới
                        open(output_json_path, 'w').close()

                        current_batch = []  # Khởi tạo batch

                        for index, row in df.iterrows():
                            frame_number = int(row['n'])
                            image_filename = f'{frame_number:03d}.jpg'
                            image_file_path = os.path.join(lxx_vxxx_path, image_filename)

                            if os.path.exists(image_file_path):
                                try:
                                    print(f"Đang thêm ảnh {image_filename} vào batch.")

                                    # Thêm ảnh vào batch cùng với thông tin cần thiết
                                    current_batch.append((image_file_path, frame_number, row))

                                    # Kiểm tra nếu batch đạt giới hạn 20 ảnh, bắt đầu xử lý batch
                                    if len(current_batch) >= max_batch_size_in_images:
                                        print(f"Batch đạt giới hạn {max_batch_size_in_images} ảnh, bắt đầu xử lý batch...")

                                        # Trích xuất vector từ ảnh gốc 1280x720
                                        images = [Image.open(item[0]).convert('RGB') for item in current_batch]
                                        features = extract_features_from_original_images(images)

                                        # Xử lý batch song song (resize chỉ để lưu vào JSON)
                                        processed_images = process_images_in_parallel(current_batch, new_size)

                                        # Tạo dữ liệu batch để lưu
                                        data_batch = []
                                        for (img_path, frame_number, row), feature, resized_image in zip(current_batch, features, processed_images):
                                            if resized_image is None:
                                                continue

                                            with open(img_path, "rb") as image_file:  # Đọc lại ảnh gốc từ đường dẫn gốc
                                                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')  # Chuyển ảnh gốc sang base64

                                            frame_info = {
                                                "keyframes_folder": keyframes_folder,  # Đổi tên để khớp với cấu trúc JSON mong muốn
                                                "video_folder": lxx_vxxx_folder,  # Thêm video_folder vào JSON
                                                "frame_number": frame_number,
                                                "image_base64": image_base64,  # Lưu ảnh gốc vào JSON
                                                "image_filename": os.path.basename(img_path),
                                                "csv_data": {
                                                    "pts_time": row['pts_time'],
                                                    "fps": row['fps'],
                                                    "frame_idx": row['frame_idx']
                                                },
                                                "vector": feature  # Đặc trưng trích xuất từ ảnh gốc
                                            }

                                            # Kiểm tra cấu trúc JSON trước khi lưu
                                            if not validate_json_structure(frame_info):
                                                print(f"JSON không hợp lệ tại frame {frame_number}, dừng chương trình.")
                                                save_checkpoint()
                                                raise SystemExit("Dừng xử lý do phát hiện JSON không hợp lệ.")

                                            data_batch.append(frame_info)

                                        # Ghi batch vào file JSON
                                        append_to_video_json(output_json_path, data_batch)

                                        print(f"Batch xử lý xong, xóa batch...")

                                        # Xóa batch hiện tại sau khi xử lý
                                        current_batch = []

                                except Exception as e:
                                    print(f"Lỗi khi xử lý ảnh {image_file_path}: {str(e)}")
                                    save_checkpoint()
                                    # Xóa file JSON chưa hoàn thành
                                    if os.path.exists(current_output_json_path):
                                        os.remove(current_output_json_path)
                                        print(f"Đã xóa file JSON chưa hoàn thành: {current_output_json_path}")
                                    raise SystemExit("Dừng chương trình do lỗi không mong muốn.")

                        # Xử lý batch cuối cùng nếu còn ảnh
                        if current_batch:
                            print(f"Xử lý batch cuối cùng chứa {len(current_batch)} ảnh...")

                            # Trích xuất vector từ ảnh gốc 1280x720
                            images = [Image.open(item[0]).convert('RGB') for item in current_batch]
                            features = extract_features_from_original_images(images)

                            # Xử lý batch song song (resize chỉ để lưu vào JSON)
                            processed_images = process_images_in_parallel(current_batch, new_size)

                            data_batch = []
                            for (img_path, frame_number, row), feature, resized_image in zip(current_batch, features, processed_images):
                                if resized_image is None:
                                    continue

                                with open(img_path, "rb") as image_file:
                                    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

                                frame_info = {
                                    "keyframes_folder": keyframes_folder,
                                    "video_folder": lxx_vxxx_folder,  # Đúng tên video folder
                                    "frame_number": frame_number,
                                    "image_base64": image_base64,
                                    "image_filename": os.path.basename(img_path),
                                    "csv_data": {
                                        "pts_time": row['pts_time'],
                                        "fps": row['fps'],
                                        "frame_idx": row['frame_idx']
                                    },
                                    "vector": feature
                                }

                                # Kiểm tra cấu trúc JSON trước khi lưu
                                if not validate_json_structure(frame_info):
                                    print(f"JSON không hợp lệ tại frame {frame_number}, dừng chương trình.")
                                    save_checkpoint()
                                    # Xóa file JSON chưa hoàn thành
                                    if os.path.exists(current_output_json_path):
                                        os.remove(current_output_json_path)
                                        print(f"Đã xóa file JSON chưa hoàn thành: {current_output_json_path}")
                                    raise SystemExit("Dừng xử lý do phát hiện JSON không hợp lệ.")

                                data_batch.append(frame_info)

                            # Ghi batch cuối cùng vào JSON
                            append_to_video_json(output_json_path, data_batch)

                            # Xóa batch hiện tại
                            current_batch = []

                        processed_points += len(df)
                        print(f"Processed {processed_points}/{total_points} points.")

                        # Cập nhật checkpoint sau khi xử lý xong thư mục con
                        processed_subs_in_folder.append(lxx_vxxx_folder)
                        processed_subfolders[keyframes_folder] = processed_subs_in_folder
                        save_checkpoint()

                        # Reset biến toàn cục
                        current_subfolder = None
                        current_output_json_path = None

                    else:
                        print(f"Không tìm thấy file CSV cho thư mục {lxx_vxxx_folder}")

# Kết thúc chương trình
print("Tất cả dữ liệu đã được xử lý hoàn tất.")
