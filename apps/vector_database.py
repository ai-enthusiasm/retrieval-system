import numpy as np
import torch
from PIL import Image
import base64
import io
import gzip
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, Filter, FieldCondition, MatchValue
from transformers import AlignProcessor, AlignModel
import logging
import os

class VectorDB:
    def __init__(self, api='http://aienthusiasm:6333', timeout=200.0, device="cuda:0" if torch.cuda.is_available() else "cpu", api_key=None):
        """Initialize QdrantImageModule with host, port, and processing mode (local or api)."""
        self.device = device
        self.api_url = api
        self.timeout = timeout
        
        # Lấy API key từ biến môi trường hoặc dùng giá trị mặc định
        self.api_key = api_key or os.environ.get('QDRANT_API_KEY')
        
        if not self.api_key:
            print("Cảnh báo: QDRANT_API_KEY không được cung cấp. Kết nối có thể không thành công.")

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
            print("✅ Kết nối thành công đến Qdrant!")
        except Exception as e:
            print(f"❌ Không thể kết nối đến Qdrant: {e}")
            raise

        # Model xử lý hình ảnh ALIGN
        self.processor_align = AlignProcessor.from_pretrained("kakaobrain/align-base")
        self.model_align = AlignModel.from_pretrained("kakaobrain/align-base").to(self.device)

        # # Model mô tả ảnh BLIP
        # self.processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        # self.model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cpu")

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
        """Tìm kiếm dữ liệu trong dataset bằng văn bản hoặc hình ảnh."""
        if not query_text:
            raise ValueError("Cần cung cấp query_text hoặc files_search để tìm kiếm")

        vector = self._get_query_vector(query_text)
        if vector is None:
            print("❌ Không thể tạo vector tìm kiếm")
            return []

        try:
            qdrant_results = self.client.search(
                collection_name="dataset",
                query_vector=vector,
                limit=150
            )
            return qdrant_results
        except Exception as e:
            print(f"❌ Lỗi khi truy vấn dataset: {e}")
            return []

    def _get_query_vector(self, query_text):
        """Xác định vector tìm kiếm dựa vào văn bản hoặc hình ảnh."""
        if query_text:
            return self.text_encode(query_text)
        return None
    
    def decode_and_decompress_image(self, base64_str, output_path):
        """Giải mã base64 và lưu ảnh."""
        try:
            image_bytes = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_bytes))
            image.save(output_path)
            logging.info(f"✅ Ảnh đã được lưu thành công tại {output_path}")
        except Exception as e:
            logging.error(f"❌ Lỗi khi giải mã và lưu ảnh: {e}")
