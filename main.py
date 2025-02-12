import cv2
import numpy as np
import pytesseract
import imutils
import os

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

while True:
    image_path = input("Введите путь к изображению (или 'exit' для выхода): ")
    if image_path.lower() == 'exit':
        break

    if not os.path.isfile(image_path):
        print("Файл не найден!")
        continue

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print("Не удалось загрузить изображение!")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_folder, "gray_image.png"), gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(os.path.join(output_folder, "gaussian.png"), blurred)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imwrite(os.path.join(output_folder, "adaptive.png"), thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_img = img.copy()
    digits_found = []

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if w * h > 100:
            roi = thresh[y:y + h, x:x + w]

            text = pytesseract.image_to_string(roi, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            
            if any(char.isdigit() for char in text):
                color = (0, 255, 0)  # Зелёный цвет для контуров с цифрами
                digits_found += 1
            else:
                color = (0, 0, 255)  # Красный цвет для контуров без цифр

            cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)

    output_image_path = os.path.join(output_folder, "detected_digits.png")
    cv2.imwrite(output_image_path, result_img)
    print(f"Изображение с выделенными цифрами сохранено в {output_image_path}")
    print(f"Найдено контуров с цифрами: {digits_found}")

print("Программа завершена")