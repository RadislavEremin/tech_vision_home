import cv2
import pyttsx3
import time

# Инициализация движка для синтеза речи
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Скорость речи
engine.setProperty('voice', 'russian')  # Если доступен русский голос, иначе используйте дефолтный

# Загрузка каскада для детекции лица (frontalface для фронтального вида)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Инициализация веб-камеры Microsoft HD CAM (индекс 1 для внешней камеры; если не сработает, попробуйте 2)
cap = cv2.VideoCapture(1)  # Изменено с 0 на 1 для дополнительной камеры

# Флаг для предотвращения повторного срабатывания
intruder_detected = False
last_detection_time = 0
cooldown = 5  # Секунды между срабатываниями

print("Камера Microsoft HD CAM запущена. Нажмите 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка захвата кадра. Проверьте подключение камеры.")
        break

    # Преобразование в grayscale для детекции
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Детекция лиц
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    current_time = time.time()

    if len(faces) > 0:
        # Рисуем прямоугольники вокруг обнаруженных лиц
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Проверяем кулдаун
        if not intruder_detected or (current_time - last_detection_time > cooldown):
            engine.say("Обнаружен нарушитель")
            engine.runAndWait()
            intruder_detected = True
            last_detection_time = current_time
            print("Обнаружено лицо нарушителя!")
    
    else:
        intruder_detected = False

    # Отображение кадра
    cv2.imshow('Детекция лица нарушителя', frame)

    # Выход по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
engine.stop()