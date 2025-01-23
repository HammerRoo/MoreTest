import cv2
import numpy as np
import pytesseract
import imutils
import os

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
config = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'

output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

image_path = input("Введите путь к изображению для распознавания текста: ")

try:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Изображение не найдено. Проверьте путь к файлу.")
except Exception as e:
    print(f"Ошибка при загрузке изображения: {e}")
    exit()

cv2.imwrite(os.path.join(output_folder, "original_image.jpg"), img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(output_folder, "gray_image.jpg"), gray)

gray = cv2.bilateralFilter(gray, 13, 15, 15)
cv2.imwrite(os.path.join(output_folder, "bilateral_filtered.jpg"), gray)

thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
cv2.imwrite(os.path.join(output_folder, "adaptive_threshold.jpg"), thresh)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Тут пока просто все контуры выделяем
for cnt in contours:
    if cv2.contourArea(cnt) > 100:
        x, y, w, h = cv2.boundingRect(cnt)

        if h > 28 and w > 10:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv2.imwrite(os.path.join(output_folder, "image_with_all_rectangles.jpg"), img)

# Тут уже выделяем только контуры с цифрами и сразу ищем ту в которой есть 8 цифр
img_with_digits_only = img.copy()
final_area = None
final_number = ""
for cnt in contours:
    if cv2.contourArea(cnt) > 100:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 28 and w > 10:
            roi = gray[y:y+h, x:x+w]
            text = pytesseract.image_to_string(roi, config=config).strip()

            if text.isdigit():
                cv2.rectangle(img_with_digits_only, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if len(text) == 8:
                    final_area = (x, y, w, h)
                    final_number = text
cv2.imwrite(os.path.join(output_folder, "image_with_digits_only.jpg"), img_with_digits_only)

# А теперь сохраняем конечный результат
if final_area:
    x, y, w, h = final_area
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(output_folder, "final_image.jpg"), img)
    print(f"Распознанный номер: {final_number}")
else:
    print("Область с 8 цифрами не найдена.")

print(f"\nПромежуточные изображения сохранены в папке: {output_folder}")