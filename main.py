import cv2
import numpy as np
import pytesseract
import imutils
import os

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

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt) > 100:
        x, y, w, h = cv2.boundingRect(cnt)

        if h > 28 and w > 10:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imwrite(os.path.join(output_folder, "output_image_with_rectangles.jpg"), img)

#         config = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
#         text = pytesseract.image_to_string(roi, config=config)

#         numbers = "".join(filter(str.isdigit, text))
#         if len(numbers) >= 8:
#             possible_texts.append(numbers)
            
#             cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             cv2.putText(img, numbers, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

#             number_found = True
#             break

# final_image_path = os.path.join(output_folder, "final_image_with_text.jpg")
# cv2.imwrite(final_image_path, img)

# if possible_texts:
#     print("\nРаспознанные номера:")
#     for num in possible_texts:
#         print(num)
# else:
#     print("Номера не найдены.")

print(f"\nПромежуточные изображения сохранены в папке: {output_folder}")