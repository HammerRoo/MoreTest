import cv2
import numpy as np
import os
import pytesseract
from collections import OrderedDict

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

output_folder = "output_images"
data_set_folder = "data_set"
got_num_folder = "got_num"
no_num_folder = "no_num"
roi_folder = os.path.join(output_folder, "roi")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(roi_folder, exist_ok=True)
os.makedirs(data_set_folder, exist_ok=True)
os.makedirs(got_num_folder, exist_ok=True)
os.makedirs(no_num_folder, exist_ok=True)

def save_to_folder(image, folder, name):
    path = os.path.join(folder, name)
    cv2.imwrite(path, image)
    print(f"Изображение сохранено: {path}")

def process_roi(roi, config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'):
    text = pytesseract.image_to_string(roi, config=config).strip()
    return text if len(text) >= 4 and text.isdigit() else None

def process_and_save_roi(roi, folder, prefix, index, text):
    if text:
        #save_to_folder(roi, folder, f"{prefix}_{index}_roi.png")
        return True
    return False

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_to_folder(gray, output_folder, "1_gray.png")

    blurred = cv2.GaussianBlur(gray, (11, 11), 7)
    save_to_folder(blurred, output_folder, "2_blurred.png")

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    save_to_folder(thresh, output_folder, "3_thresh.png")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    save_to_folder(opened, output_folder, "4_opened.png")

    dilated_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(opened, dilated_kernel, iterations=4)
    save_to_folder(dilated, output_folder, "5_dilated.png")

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_image = image.copy()

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    save_to_folder(output_image, output_folder, "6_result.jpg")

    return dilated

def find_and_draw_digits(image, processed_image):
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_numbers = []
    got_num = image.copy()
    no_num = image.copy()

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area < 1000:
            continue

        roi = image[y:y + h, x:x + w]

        text_clean = process_roi(roi)
        if process_and_save_roi(roi, roi_folder, "1_original", i, text_clean):
            detected_numbers.append(text_clean)
            continue

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        text_gray = process_roi(roi_gray)
        if process_and_save_roi(roi_gray, roi_folder, "2_gray", i, text_gray):
            detected_numbers.append(text_gray)
            continue

        binary_roi = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 9)
        text_binary = process_roi(binary_roi)
        if process_and_save_roi(binary_roi, roi_folder, "3_binary", i, text_binary):
            detected_numbers.append(text_binary)
            continue

        dilated_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated_roi = cv2.dilate(binary_roi, dilated_kernel, iterations=2)
        text_dilate = process_roi(dilated_roi)
        if process_and_save_roi(dilated_roi, roi_folder, "5_dilated", i, text_dilate):
            detected_numbers.append(text_dilate)
            continue

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, close_kernel, iterations=5)
        text_close = process_roi(morph_roi)
        if process_and_save_roi(morph_roi, roi_folder, "6_morph_close", i, text_close):
            detected_numbers.append(text_close)
            continue

        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph_roi = cv2.morphologyEx(morph_roi, cv2.MORPH_OPEN, open_kernel, iterations=1)
        text_open = process_roi(morph_roi)
        if process_and_save_roi(morph_roi, roi_folder, "7_morph_open", i, text_open):
            detected_numbers.append(text_open)
            continue

    # Сохраняем итоговые изображения
    if detected_numbers:
        save_to_folder(got_num, got_num_folder, f"got_num_{i}.png")
    else:
        save_to_folder(no_num, no_num_folder, f"no_num_{i}.png")

    return list(OrderedDict.fromkeys(detected_numbers))

def process_video(video_path, process_frame_func):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Ошибка загрузки видео. Проверьте путь и попробуйте снова.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    frame_count = 0
    paused = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Видео закончилось или произошла ошибка чтения.")
            break
        
        frame_count += 1

        cv2.imshow("Video", frame)
        
        key = cv2.waitKey(delay if not paused else 0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            process_frame_func(frame)
        elif key == ord('p'):
            paused = not paused
            if paused:
                print("Видео на паузе. Нажмите 'p', чтобы продолжить.")
            else:
                print("Воспроизведение возобновлено.")
                
    cap.release()
    cv2.destroyAllWindows()

def main():
    while True:
        print("Выберите режим:")
        print("1. Кадр: выделение области")
        print("2. Кадр: поиск и выделение номера")
        print("3. Выход")
        choice = input("Введите номер режима: ")
        
        if choice == '1':
            video_path = input("Введите путь к видео: ")
            process_video(video_path, preprocess_image)
        elif choice == '2':
            video_path = input("Введите путь к видео: ")
            process_video(video_path, lambda frame: find_and_draw_digits(frame, preprocess_image(frame)))
        elif choice == '3':
            print("Выход из программы.")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")
            
if __name__ == "__main__":
    main()