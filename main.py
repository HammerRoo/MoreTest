import cv2
import numpy as np
import os
import pytesseract
from collections import OrderedDict

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

output_folder = "output_images"
raw_data_set_folder = "raw_data_set"
prep_data_set_folder = "prep_data_set"
cont_data_set_folder = "cont_data_set"
got_num_folder = "got_num"
no_num_folder = "no_num"
roi_folder = os.path.join(output_folder, "roi")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(roi_folder, exist_ok=True)
os.makedirs(raw_data_set_folder, exist_ok=True)
os.makedirs(prep_data_set_folder, exist_ok=True)
os.makedirs(cont_data_set_folder, exist_ok=True)
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

    blurred = cv2.GaussianBlur(gray, (9, 9), 7)
    save_to_folder(blurred, output_folder, "2_blurred.png")

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 5) # 31.7
    save_to_folder(thresh, output_folder, "3_thresh.png")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    save_to_folder(opened, output_folder, "4_opened.png")

    dilated_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(opened, dilated_kernel, iterations=4)
    save_to_folder(dilated, output_folder, "5_dilated.png")

    return dilated

def find_and_draw_digits(raw_image, processed_image, image_counter, save_results=False):
    all_image = raw_image.copy()
    output_image = raw_image.copy()

    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(all_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if save_results:
        save_to_folder(all_image, cont_data_set_folder, f"{image_counter}.png")

    detected_numbers = []

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area < 1000:
            continue

        roi = raw_image[y:y + h, x:x + w]

        text_clean = process_roi(roi)
        if process_and_save_roi(roi, roi_folder, "1_original", i, text_clean):
            detected_numbers.append(text_clean)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            continue

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        text_gray = process_roi(roi_gray)
        if process_and_save_roi(roi_gray, roi_folder, "2_gray", i, text_gray):
            detected_numbers.append(text_gray)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            continue

        binary_roi = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
        text_binary = process_roi(binary_roi)
        if process_and_save_roi(binary_roi, roi_folder, "3_binary", i, text_binary):
            detected_numbers.append(text_binary)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            continue

        dilated_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated_roi = cv2.dilate(binary_roi, dilated_kernel, iterations=2)
        text_dilate = process_roi(dilated_roi)
        if process_and_save_roi(dilated_roi, roi_folder, "5_dilated", i, text_dilate):
            detected_numbers.append(text_dilate)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            continue

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, close_kernel, iterations=5)
        text_close = process_roi(morph_roi)
        if process_and_save_roi(morph_roi, roi_folder, "6_morph_close", i, text_close):
            detected_numbers.append(text_close)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            continue

        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph_roi = cv2.morphologyEx(morph_roi, cv2.MORPH_OPEN, open_kernel, iterations=1)
        text_open = process_roi(morph_roi)
        if process_and_save_roi(morph_roi, roi_folder, "7_morph_open", i, text_open):
            detected_numbers.append(text_open)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            continue

    if detected_numbers:
        save_to_folder(output_image, got_num_folder, f"{image_counter}.png")
        print(f"Найдены номера в изображении {image_counter}: {detected_numbers}")
    else:
        save_to_folder(output_image, no_num_folder, f"{image_counter}.png")
        print(f"Номера не найдены в изображении {image_counter}.")

    return list(OrderedDict.fromkeys(detected_numbers))

def process_video(video_path, save_raw=False, save_prep=False, prep=False):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Ошибка загрузки видео. Проверьте путь и попробуйте снова.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    frame_count = 0
    paused = False

    save_count_1 = 1
    save_count_2 = 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Видео закончилось или произошла ошибка чтения.")
            break
        
        cv2.imshow("Video", frame)
        
        key = cv2.waitKey(delay if not paused else 0) & 0xFF
        
        if key == ord('q'):
            break

        elif key == ord(' '):
            if save_raw:
                print("Сохранение оригинальных кадров включено")
                save_to_folder(frame, raw_data_set_folder, f"{save_count_1}.png")
                save_count_1 += 1
            else:
                print("Сохранение оригинальных кадров отключено")
            
            if prep:
                print("Предобработка кадров включена")
                processed_image = preprocess_image(frame)
                if save_prep:
                    print("Сохранение предобработанных кадров включена")
                    save_to_folder(processed_image, prep_data_set_folder, f"{save_count_2}.png")
                    save_count_2 += 1
                else:
                    print("Сохранение предобработанных кадров выключено")
            else:
                print("Предобработка кадров отключена")

        elif key == ord('p'):
            paused = not paused
            if paused:
                print("Видео на паузе. Нажмите 'p', чтобы продолжить.")
            else:
                print("Воспроизведение возобновлено.")
                
    cap.release()
    cv2.destroyAllWindows()

def process_prep_images():
    raw_images = sorted(os.listdir(raw_data_set_folder), key=lambda x: int(x.split('.')[0]))
    prep_images = sorted(os.listdir(prep_data_set_folder), key=lambda x: int(x.split('.')[0]))
    
    if len(raw_images) != len(prep_images):
        print("Ошибка: количество файлов в raw_data_set и prep_data_set не совпадает.")
        return
    
    image_counter = 1

    for raw_name, prep_name in zip(raw_images, prep_images):
        raw_path = os.path.join(raw_data_set_folder, raw_name)
        prep_path = os.path.join(prep_data_set_folder, prep_name)
        
        raw_image = cv2.imread(raw_path)
        prep_image = cv2.imread(prep_path, cv2.IMREAD_GRAYSCALE)
        
        if raw_image is None or prep_image is None:
            print(f"Ошибка загрузки изображений: {raw_name} или {prep_name}")
            continue
        
        print(f"Обработка изображения {image_counter}: {raw_name}")
        detected_numbers = find_and_draw_digits(raw_image, prep_image, image_counter, False)
        
        user_input = input("Нажмите Enter для продолжения или 'q' для выхода: ")
        if user_input.lower() == 'q':
            break
        
        image_counter += 1

def main():
    while True:
        print("Выберите режим:")
        print("1. Кадр: первичная предобработка")
        print("2. Кадр: поиск и выделение номера")
        print("3. Выход")
        choice = input("Введите номер режима: ")
        
        if choice == '1':
            video_path = input("Введите путь к видео: ")
            process_video(video_path, save_raw=False, save_prep=False, prep=False)
        elif choice == '2':
            process_prep_images()
        elif choice == '3':
            print("Выход из программы.")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")
            
if __name__ == "__main__":
    main()