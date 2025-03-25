from paddleocr import PaddleOCR

ocr_model = PaddleOCR(use_angle_cls=False, lang='ru', show_log=False)

def process_roi_nn(roi):
    result = ocr_model.ocr(roi, cls=False)
    texts = [line[1][0] for line in result[0] if line[1][0].isdigit()]
    return [text for text in texts if len(text) == 8]