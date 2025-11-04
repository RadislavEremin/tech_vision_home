import cv2
import pyttsx3
import time
import os
import numpy as np  # Для обработки массивов
from datetime import datetime
from ultralytics import YOLO  # Для детекции объектов (кошек) с YOLOv8

# Инициализация движка для синтеза речи
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Скорость речи
engine.setProperty('voice', 'russian')  # Если доступен русский голос, иначе используйте дефолтный

# Загрузка каскадов для детекции лица (фронтальное и профиль)
face_cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Загрузка модели YOLOv8 для детекции объектов (включая кошек, класс 'cat' = 15)
model = YOLO('yolov8n.pt')  # Nano-модель, быстрая; скачается автоматически при первом запуске

# НОВОЕ: Инициализация face recognizer (LBPH — простой и быстрый)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer_file = 'face_recognizer.yml'
known_faces_dir = 'known_faces'  # Папка с фото известного человека

# НОВОЕ: Функция для обучения recognizer на фото
def train_face_recognizer():
    if not os.path.exists(known_faces_dir):
        print(f"ОШИБКА: Папка {known_faces_dir} не найдена! Создайте её и добавьте фото.")
        return False
    
    faces = []
    labels = []
    label_id = 0  # Label 0 для известного человека
    
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(known_faces_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Детекция лица в обучающем фото (используем фронтальный cascade)
            faces_detected = face_cascade_frontal.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces_detected:
                roi = gray[y:y+h, x:x+w]  # ROI лица
                # ИСПРАВЛЕНИЕ: Resize на фиксированный размер для consistency
                roi = cv2.resize(roi, (100, 100))
                faces.append(roi)
                labels.append(label_id)
                print(f"Обучено на фото: {filename}, ROI размер {roi.shape}")
    
    if len(faces) == 0:
        print("ОШИБКА: Ни одного лица не найдено в фото! Добавьте фронтальные фото.")
        return False
    
    # Обучение
    recognizer.train(faces, np.array(labels))
    recognizer.save(recognizer_file)
    print(f"Обучено на {len(faces)} лицах. Модель сохранена: {recognizer_file}")
    return True

# НОВОЕ: Загрузка обученной модели (если файл существует)
trained = False
if os.path.exists(recognizer_file):
    recognizer.read(recognizer_file)
    print("Загружена обученная модель из файла.")
    trained = True
else:
    print("Обучаю модель на фото из папки 'known_faces'...")
    trained = train_face_recognizer()

if not trained:
    print("ПРЕДУПРЕЖДЕНИЕ: Face recognition отключено. Используется только детекция лиц.")

# Инициализация веб-камеры Microsoft HD CAM (индекс 1; если не сработает, попробуйте 2)
cap = cv2.VideoCapture(1)

# Флаги и таймеры для подтверждения обнаружения (отдельно для лиц и кошек)
face_detected = False
cat_detected = False
face_first_time = 0
cat_first_time = 0
face_last_time = 0
cat_last_time = 0
confirmation_delay = 3  # Секунды для подтверждения
cooldown = 10  # Секунды между срабатываниями
face_last_alert = 0
cat_last_alert = 0

# НОВОЕ: Для specific person (известного лица)
specific_person_detected = False
specific_first_time = 0
specific_last_time = 0
specific_person_alert = 0  # Cooldown для сообщений о "своём" лице (опционально)

# Папка для скриншотов
screenshots_dir = 'screenshots'
if not os.path.exists(screenshots_dir):
    os.makedirs(screenshots_dir)

print("Камера Microsoft HD CAM запущена. Нажмите 'q' для выхода.")
if trained:
    print("Face recognition включено: label 0 = 'Вы' (известный), другие = 'Нарушитель'.")
print("ОТЛАДКА: Если кошка не детектируется, проверьте консоль на 'DET: Cats: N' (N > 0 значит видит кошку).")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка захвата кадра. Проверьте подключение камеры.")
        break

    # Преобразование в grayscale для детекции лиц
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Детекция фронтальных и профильных лиц
    faces_frontal = face_cascade_frontal.detectMultiScale(gray, 1.1, 4)
    faces_profile = face_cascade_profile.detectMultiScale(gray, 1.1, 4)
    all_faces = list(faces_frontal) + list(faces_profile)  # Список tuple'ов (x,y,w,h)
    
    # НОВОЕ: Распознавание для детектированных лиц (если модель обучена)
    recognized_faces = []  # Список (label, conf) для каждого лица в all_faces
    if trained and len(all_faces) > 0:
        for face in all_faces:
            x, y, w, h = face
            roi = gray[y:y+h, x:x+w]  # ROI лица
            if roi.size > 0:  # Проверка на пустой ROI
                # ИСПРАВЛЕНИЕ: Resize на 100x100 (как в обучении)
                roi = cv2.resize(roi, (100, 100))
                label, confidence = recognizer.predict(roi)
                recognized_faces.append((label, confidence))
                print(f"ОТЛАДКА: Face recognized: label {label}, conf {confidence:.2f}")
            else:
                recognized_faces.append((1, 999))  # Фейк: unknown с высоким conf
    
    # Детекция объектов с YOLO (включая кошек)
    results = model(frame, verbose=False)  # Детекция на кадре
    cats_detected = []  # Список bbox для кошек
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls) == 15:  # Класс 15 = 'cat' в COCO датасете
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cats_detected.append((x1, y1, x2 - x1, y2 - y1))
    
    current_time = time.time()

    # ОТЛАДКА: Вывод количества детектированных кошек (каждые 30 кадров, чтобы не спамить)
    if int(current_time * 30) % 30 == 0:  # Примерно каждую секунду при 30 FPS
        print(f"ОТЛАДКА: DET: Cats: {len(cats_detected)} | Faces: {len(all_faces)} | Recognized: {len(recognized_faces)}")

    # НОВОЕ: Логика для specific person (известного лица)
    is_specific_person_present = False
    if trained:
        for (label, conf) in recognized_faces:
            if label == 0 and conf < 100:  # Label 0 = известный, conf < 100 = хорошее совпадение
                is_specific_person_present = True
                break
    
    if is_specific_person_present:
        if not specific_person_detected:
            specific_first_time = current_time
            specific_person_detected = True
        specific_last_time = current_time
        
        # Опциональный alert для "своего" лица (можно закомментировать)
        if (current_time - specific_first_time >= confirmation_delay) and \
           (current_time - specific_person_alert >= cooldown * 2):  # Редко
            engine.say("Это вы, всё в порядке")
            engine.runAndWait()
            print("Подтверждено: Распознано известное лицо!")
            specific_person_alert = current_time
    else:
        specific_person_detected = False
        specific_first_time = 0
        specific_last_time = 0

    # Логика для лиц (модифицирована: alert только если НЕ specific person)
    unknown_intruder = False
    if len(all_faces) > 0:
        # ИСПРАВЛЕНИЕ: Рисуем в одном цикле, определяя тип по индексу
        num_frontal = len(faces_frontal)
        for i, face in enumerate(all_faces):
            x, y, w, h = face
            is_frontal = i < num_frontal  # Первые — фронтальные
            base_color = (255, 0, 0) if is_frontal else (0, 0, 255)  # Синий/красный базово
            color = base_color
            label_text = "Face" if is_frontal else "Profile"
            
            if trained and i < len(recognized_faces):
                label, conf = recognized_faces[i]
                print(f"ОТЛАДКА: {label_text} - label {label}, conf {conf:.2f}")  # Доп. отладка
                if label == 0 and conf < 100:
                    color = (0, 255, 0)  # Зелёный для известного
                    label_text = f"You {int(conf)}"
                else:
                    color = (0, 0, 255)  # Красный для неизвестного
                    unknown_intruder = True
                    label_text = f"Unknown {int(conf)}"
            else:
                unknown_intruder = True
                label_text = "Unknown"
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        if not face_detected:
            face_first_time = current_time
            face_detected = True
        face_last_time = current_time
        
        # Подтверждение только для неизвестного
        if unknown_intruder and (current_time - face_first_time >= confirmation_delay) and \
           (current_time - face_last_alert >= cooldown):
            engine.say("Обнаружен нарушитель")
            engine.runAndWait()
            print("Подтверждено: Обнаружено неизвестное лицо!")
            
            # Скриншот для неизвестного
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(screenshots_dir, f'intruder_{timestamp}.jpg')
            cv2.imwrite(screenshot_path, frame)
            print(f"Скриншот сохранен: {screenshot_path}")
            
            face_last_alert = current_time
            face_first_time = current_time  # Сброс для следующего
    
    else:
        face_detected = False
        face_first_time = 0
        face_last_time = 0

    # Логика для кошек (без изменений)
    if len(cats_detected) > 0:
        # Рисуем зеленые прямоугольники вокруг кошек
        for (x, y, w, h) in cats_detected:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Cat', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        if not cat_detected:
            cat_first_time = current_time
            cat_detected = True
        cat_last_time = current_time
        
        # Подтверждение для кошки
        if (current_time - cat_first_time >= confirmation_delay) and \
           (current_time - cat_last_alert >= cooldown):
            engine.say("Обнаружена глупая дебильная вообще тупая кошка")
            engine.runAndWait()
            print("Подтверждено: Обнаружена кошка!")
            
            # Скриншот для кошки
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(screenshots_dir, f'cat_{timestamp}.jpg')
            cv2.imwrite(screenshot_path, frame)
            print(f"Скриншот сохранен: {screenshot_path}")
            
            cat_last_alert = current_time
            cat_first_time = current_time  # Сброс для следующего
    
    else:
        cat_detected = False
        cat_first_time = 0
        cat_last_time = 0

    # Отображение кадра
    cv2.imshow('Детекция лица/кошки/человека', frame)

    # Выход по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
engine.stop()