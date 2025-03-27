import easyocr
from paddleocr import PaddleOCR
from mmocr.apis import TextRecInferencer
import yaml
import re
import cv2

class OCRProcessor:
    def __init__(self, config_path="config.yml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.engine = self.config['ocr']['active_engine']
        self.min_len, self.max_len = self.config['ocr']['number_length']
        
        self._init_engine()
    
    def _init_engine(self):
        if self.engine == "easyocr":
            self.reader = easyocr.Reader(
                self.config['ocr']['easyocr']['languages'],
                gpu=self.config['ocr']['easyocr']['gpu'],
                model_storage_dir=self.config['ocr']['easyocr']['model_storage']
            )
        elif self.engine == "paddleocr":
            self.ocr = PaddleOCR(
                lang=self.config['ocr']['paddleocr']['lang'],
                use_gpu=self.config['ocr']['paddleocr']['use_gpu'],
                use_angle_cls=self.config['ocr']['paddleocr']['use_angle_cls']
            )
        elif self.engine == "mmocr":
            self.recognizer = TextRecInferencer(
                model=self.config['ocr']['mmocr']['model'],
                device=self.config['ocr']['mmocr']['device']
            )
    
    def extract_numbers(self, image):
        if self.engine == "easyocr":
            return self._easyocr_process(image)
        elif self.engine == "paddleocr":
            return self._paddleocr_process(image)
        elif self.engine == "mmocr":
            return self._mmocr_process(image)
    
    def _validate_number(self, text):
        return bool(re.fullmatch(rf'\d{{{self.min_len},{self.max_len}}}', text.strip()))
    
    def _easyocr_process(self, image):
        results = self.reader.readtext(
            image,
            allowlist='0123456789',
            text_threshold=0.6,
            paragraph=True
        )
        return [res[1] for res in results if self._validate_number(res[1])]
    
    def _paddleocr_process(self, image):
        result = self.ocr.ocr(image, cls=True)
        numbers = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                if self._validate_number(text):
                    numbers.append(text)
        return numbers
    
    def _mmocr_process(self, image):
        result = self.recognizer(image)['predictions']
        return [res['text'] for res in result 
               if self._validate_number(res['text']) and res['confidence'] > 0.5]