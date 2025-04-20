import numpy as np
import torch
from PIL import Image
import base64
import io
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, Filter, FieldCondition, MatchValue
from transformers import AlignProcessor, AlignModel
import logging

class VectorDB:
    def __init__(self, api='http://aienthusiasm:6333', timeout=200.0, device="cuda:0" if torch.cuda.is_available() else "cpu", api_key= None):
        """Initialize QdrantImageModule with host, port, and processing mode (local or api)."""
        self.device = device
        self.api_url = api
        self.timeout = timeout
        self.api_key = api_key

        # Khởi tạo Qdrant Client
        self.client = QdrantClient(
            url=self.api_url,
            api_key=self.api_key,
            timeout=self.timeout,
            https=False  # Bắt buộc nếu server không dùng SSL
        )

        # Kiểm tra kết nối
        try:
            self.client.get_collections()
            print("Kết nối thành công đến Qdrant!")
        except Exception as e:
            print(f"Không thể kết nối đến Qdrant: {e}")
            raise

        # Model xử lý hình ảnh ALIGN
        self.processor_align = AlignProcessor.from_pretrained("kakaobrain/align-base")
        self.model_align = AlignModel.from_pretrained("kakaobrain/align-base").to(self.device)

    def text_encode(self, text):
        """Mã hóa văn bản thành vector."""
        processed_text = self.processor_align(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.model_align.get_text_features(
                input_ids=processed_text['input_ids'],
                attention_mask=processed_text['attention_mask']
            ).cpu().numpy().flatten()
        return text_features.tolist()

    def query_dataset(self, query_text=None):
        """Tìm kiếm dữ liệu trong dataset bằng văn bản."""
        if not query_text:
            raise ValueError("Cần cung cấp query_text để tìm kiếm")

        vector = self.text_encode(query_text)
        if vector is None:
            print("Không thể tạo vector tìm kiếm")
            return []

        try:
            qdrant_results = self.client.search(
                collection_name="dataset",
                query_vector=vector,
                limit=150
            )
            return qdrant_results
        except Exception as e:
            print(f"Lỗi khi truy vấn dataset: {e}")
            return []
    
    def decode_and_decompress_image(self, base64_str, output_path):
        """Giải mã base64 và lưu ảnh."""
        try:
            image_bytes = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_bytes))
            image.save(output_path)
            logging.info(f"Ảnh đã được lưu thành công tại {output_path}")
        except Exception as e:
            logging.error(f"Lỗi khi giải mã và lưu ảnh: {e}")


# import numpy as np
# import torch
# from PIL import Image
# import base64
# import io
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import PointStruct, Distance, Filter, FieldCondition, MatchValue
# from transformers import AlignProcessor, AlignModel
# import logging
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from pathlib import Path
# from typing import List, Dict, Union, Tuple
# import os

# # Cấu hình logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

# class VectorDB:
#     def __init__(self, api='http://aienthusiasm:6333', timeout=200.0, 
#                  device="cuda:0" if torch.cuda.is_available() else "cpu", 
#                  api_key=None):
#         """Initialize QdrantImageModule with host, port, and processing mode (local or api)."""
#         self.device = device
#         self.api_url = api
#         self.timeout = timeout
#         self.api_key = api_key

#         # Khởi tạo Qdrant Client
#         self.client = QdrantClient(
#             url=self.api_url,
#             api_key=self.api_key,
#             timeout=self.timeout,
#             https=False  # Bắt buộc nếu server không dùng SSL
#         )

#         # Kiểm tra kết nối
#         try:
#             self.client.get_collections()
#             logging.info("Kết nối thành công đến Qdrant!")
#         except Exception as e:
#             logging.error(f"Không thể kết nối đến Qdrant: {e}")
#             raise

#         # Model xử lý hình ảnh ALIGN
#         self.processor_align = AlignProcessor.from_pretrained("kakaobrain/align-base")
#         self.model_align = AlignModel.from_pretrained("kakaobrain/align-base").to(self.device)

#     def text_encode(self, text: str) -> List[float]:
#         """Mã hóa văn bản thành vector."""
#         processed_text = self.processor_align(text=text, return_tensors="pt").to(self.device)
#         with torch.no_grad():
#             text_features = self.model_align.get_text_features(
#                 input_ids=processed_text['input_ids'],
#                 attention_mask=processed_text['attention_mask']
#             ).cpu().numpy().flatten()
#         return text_features.tolist()

#     def query_dataset(self, query_text: str = None) -> List[Dict]:
#         """Tìm kiếm dữ liệu trong dataset bằng văn bản."""
#         if not query_text:
#             raise ValueError("Cần cung cấp query_text để tìm kiếm")

#         vector = self.text_encode(query_text)
#         if vector is None:
#             logging.error("Không thể tạo vector tìm kiếm")
#             return []

#         try:
#             qdrant_results = self.client.search(
#                 collection_name="dataset",
#                 query_vector=vector,
#                 limit=50
#             )
#             return qdrant_results
#         except Exception as e:
#             logging.error(f"Lỗi khi truy vấn dataset: {e}")
#             return []

#     @staticmethod
#     def decode_and_decompress_image_single(base64_str: str, output_path: str) -> str:
#         """Giải mã và lưu một ảnh từ base64 (dùng trong tiến trình con)."""
#         try:
#             # Validate base64 string
#             if not base64_str:
#                 return f"❌ Lỗi ảnh {output_path}: Chuỗi base64 trống"

#             # Remove base64 header if exists
#             if ',' in base64_str:
#                 base64_str = base64_str.split(',')[1]

#             # Add padding if needed
#             padding = 4 - (len(base64_str) % 4)
#             if padding != 4:
#                 base64_str += '=' * padding

#             # Create output directory if not exists
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)

#             # Decode and save image
#             image_bytes = base64.b64decode(base64_str)
#             image = Image.open(io.BytesIO(image_bytes))
            
#             # Verify image data
#             image.verify()
#             image = Image.open(io.BytesIO(image_bytes))  # Reopen after verify
            
#             # Get format from image if possible, fallback to path suffix
#             format = image.format
#             if not format:
#                 format = Path(output_path).suffix[1:].upper() or 'JPEG'
            
#             image.save(output_path, format=format)
#             return f"✅ Đã lưu: {output_path}"
            
#         except Exception as e:
#             return f"❌ Lỗi ảnh {output_path}: {e}"

#     def decode_images_parallel(self, base64_list: List[str], output_paths: List[str], 
#                              max_workers: int = 36) -> List[str]:
#         """
#         Xử lý nhiều ảnh base64 song song và lưu ra file.
        
#         Args:
#             base64_list: Danh sách chuỗi base64
#             output_paths: Danh sách đường dẫn output tương ứng
#             max_workers: Số lượng worker tối đa (mặc định là None - tự động theo CPU)
        
#         Returns:
#             List[str]: Danh sách kết quả xử lý cho từng ảnh
#         """
#         if len(base64_list) != len(output_paths):
#             raise ValueError("Số lượng base64 và output_path phải bằng nhau")

#         results = []
        
#         # Sử dụng ProcessPoolExecutor để xử lý song song
#         with ProcessPoolExecutor(max_workers=max_workers) as executor:
#             futures = [
#                 executor.submit(self.decode_and_decompress_image_single, b64, path)
#                 for b64, path in zip(base64_list, output_paths)
#             ]
            
#             # Thu thập kết quả theo thứ tự hoàn thành
#             for future in as_completed(futures):
#                 try:
#                     result = future.result()
#                     results.append(result)
#                 except Exception as e:
#                     results.append(f"❌ Lỗi không xác định: {str(e)}")

#         return results

#     def decode_and_decompress_image(self, base64_str: str, output_path: str) -> bool:
#         """
#         Giải mã một ảnh base64 và lưu (phiên bản đồng bộ).
        
#         Args:
#             base64_str: Chuỗi base64
#             output_path: Đường dẫn lưu file
            
#         Returns:
#             bool: True nếu thành công, False nếu thất bại
#         """
#         result = self.decode_and_decompress_image_single(base64_str, output_path)
#         success = result.startswith("✅")
#         if not success:
#             logging.error(result)
#         return success
