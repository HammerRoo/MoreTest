import cv2
import numpy as np
import pytesseract
import imutils
import os

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
config = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'

output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

while True:
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

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imwrite(os.path.join(output_folder, "adaptive_threshold.jpg"), thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Тут пока просто все контуры выделяем
    img_all_rectangles = img.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)

            if h > 24 and w > 10:
                cv2.rectangle(img_all_rectangles, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imwrite(os.path.join(output_folder, "image_with_all_rectangles.jpg"), img_all_rectangles)

    # Тут уже выделяем только контуры с цифрами и сразу ищем ту в которой есть 8 цифр
    img_with_digits_only = img.copy()
    found_digits = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > 24 and w > 10:
                roi = gray[y:y+h, x:x+w]
                text = pytesseract.image_to_string(roi, config=config).strip()

                if text.isdigit():
                    found_digits.append((x, y, w, h, text))
                    cv2.rectangle(img_with_digits_only, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imwrite(os.path.join(output_folder, "image_with_digits_only.jpg"), img_with_digits_only)


    final_area = None
    final_number = ""

    # В случае если нашли область с 8-ю цифрами
    for x, y, w, h, text in found_digits:
        if len(text) == 8:
            final_area = (x, y, w, h)
            final_number = text
            break

    # В случае если область с 8-ю цифрами не найдена
    if not final_number and len(found_digits) >= 8:
        found_digits.sort(key=lambda d: (d[1], d[0]))
        grouped_digits = found_digits[:8]

        x_min = min(d[0] for d in  grouped_digits)
        y_min = min(d[1] for d in  grouped_digits)
        x_max = max(d[0] + d[2] for d in  grouped_digits)
        y_max = max(d[1] + d[3] for d in  grouped_digits)

        final_area = (x_min, y_min, x_max - x_min, y_max - y_min)
        final_number = "".join(d[4] for d in grouped_digits)

    # А теперь сохраняем конечный результат
    if final_area:
        x, y, w, h = final_area
        img_final = img.copy()

        cv2.rectangle(img_final, (x, y), (x + w, y + h), (255, 0, 0), 2)
        text_size = int(max(w, h) * 0.2)
        cv2.putText(img_final, final_number, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, text_size / 30, (255, 0, 0), 2)

        cv2.imwrite(os.path.join(output_folder, "final_image.jpg"), img_final)
        print(f"Распознанный номер: {final_number}")
    else:
        print("Область с 8 цифрами не найдена.")

print(f"\nПромежуточные изображения сохранены в папке: {output_folder}")
print("Программа завершена")