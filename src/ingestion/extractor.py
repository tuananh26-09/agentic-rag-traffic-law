import os
import re
import fitz
import numpy as np
import gc
import cv2
from PIL import Image
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

class OCRExtractor:
    def __init__(self):
        print("Đang khởi động PaddleOCR (Chuyên gia Bố cục)...")
        self.detector = PaddleOCR(use_angle_cls=True, lang='vi', rec=False, show_log=False)
        
        print("Đang khởi động VietOCR (Chuyên gia Tiếng Việt)...")
        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = 'cpu'
        self.recognizer = Predictor(config)

    def extract_text_from_pdf(self, pdf_path):  
        print(f" Đang xử lý Digital PDF: {os.path.basename(pdf_path)}")
        doc = fitz.open(pdf_path)
        full_text = "" 
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = ""
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: b[1]) 
            
            for b in blocks:
                if b[6] == 0: 
                    block_text = b[4].strip()
                    if block_text.isdigit() and len(block_text) < 5:
                        continue
                    page_text += block_text + "\n\n"
            
            if not page_text.strip():
                print(f" Trang {page_num + 1} không có chữ, kích hoạt OCR...")
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                page_text = self.extract_text_from_image_bytes(img_bytes) + "\n\n"
                
            if full_text and full_text.strip() and full_text.strip()[-1] not in ['.', ';', ':']:
                full_text = full_text.rstrip() + " " + page_text.lstrip()
            else:
                full_text += page_text

        return [{
            "text": full_text,
            "metadata": {"source": os.path.basename(pdf_path), "type": "pdf"}
        }]

    def extract_text_from_image(self, img_path):
        print(f" Đang xử lý Hình ảnh Hybrid (Paddle+VietOCR): {os.path.basename(img_path)}")
        img = cv2.imread(img_path)
        if img is None:
            return []
        
        boxes = self.detector.ocr(img_path, det=True, rec=False, cls=True)
        text_lines = []
        
        if boxes and boxes[0]:
            sorted_boxes = sorted(boxes[0], key=lambda x: x[0][1])
            for box in sorted_boxes:
                box = np.array(box).astype(np.int32)
                x_min, x_max = np.min(box[:, 0]), np.max(box[:, 0])
                y_min, y_max = np.min(box[:, 1]), np.max(box[:, 1])
                
                pad = 4
                y_min, y_max = max(0, y_min - pad), min(img.shape[0], y_max + pad)
                x_min, x_max = max(0, x_min - pad), min(img.shape[1], x_max + pad)
                
                cropped_img = img[y_min:y_max, x_min:x_max]
                if cropped_img.size == 0 or cropped_img.shape[0] < 5 or cropped_img.shape[1] < 5:
                    continue
                    
                cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cropped_img_rgb)
                
                text = self.recognizer.predict(pil_img)
                clean_text = text.strip()
                
                if clean_text:
                    if clean_text.isdigit() and len(clean_text) < 4:
                        continue
                    if clean_text.endswith("Thuận"):
                        clean_text = clean_text[:-5] + ":"
                    elif clean_text.endswith("thuận"):
                        clean_text = clean_text[:-5] + ","
                    text_lines.append(clean_text)
                
        gc.collect()
        
        healed_lines = []
        current_sentence = ""
        
        for line in text_lines:
            if not current_sentence:
                current_sentence = line
            else:
                is_new_paragraph = re.match(r'^(Điều\s+\d+|[a-zđ]\)|[0-9]+\.)', line, re.IGNORECASE)
                if is_new_paragraph:
                    healed_lines.append(current_sentence)
                    current_sentence = line
                elif current_sentence[-1] not in ['.', ':', ';']:
                    current_sentence += " " + line
                else:
                    healed_lines.append(current_sentence)
                    current_sentence = line
                    
        if current_sentence:
            healed_lines.append(current_sentence)
        
        final_text = "\n\n".join(healed_lines)
        return [{
            "text": final_text,
            "metadata": {"source": os.path.basename(img_path), "page": 1, "type": "image"}
        }]

    def extract_text_from_image_bytes(self, img_bytes):
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        boxes = self.detector.ocr(img, det=True, rec=False, cls=True)
        text_lines = []
        
        if boxes and boxes[0]:
            sorted_boxes = sorted(boxes[0], key=lambda x: x[0][1])
            for box in sorted_boxes:
                box = np.array(box).astype(np.int32)
                x_min, x_max = np.min(box[:, 0]), np.max(box[:, 0])
                y_min, y_max = np.min(box[:, 1]), np.max(box[:, 1])
                
                cropped_img = img[y_min:y_max, x_min:x_max]
                if cropped_img.size == 0:
                    continue
                    
                pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                text_lines.append(self.recognizer.predict(pil_img).strip())
                
        return " ".join(text_lines)