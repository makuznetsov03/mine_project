import os
import imageio
import glob

print("Создание GIF-анимации из кадров...")

# Проверяем наличие кадров
frames_dir = "animation_frames"
if not os.path.exists(frames_dir):
    print(f"Ошибка: директория {frames_dir} не существует!")
    exit(1)

# Получаем список файлов кадров, сортируем по имени
frame_files = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.png')))

if not frame_files:
    print(f"Ошибка: в директории {frames_dir} не найдены кадры!")
    exit(1)

print(f"Найдено {len(frame_files)} кадров")

# Загружаем кадры
images = []
for filename in frame_files:
    images.append(imageio.imread(filename))

# Создаем GIF, уменьшаем fps для более плавной анимации
print("Создание GIF-анимации...")
imageio.mimsave('workers_animation.gif', images, fps=10)

print("GIF-анимация успешно создана: workers_animation.gif")

# Также создаем видео с помощью imageio, если это поддерживается
try:
    print("Попытка создания видео MP4...")
    imageio.mimsave('workers_animation.mp4', images, fps=20)
    print("Видео успешно создано: workers_animation.mp4")
except Exception as e:
    print(f"Не удалось создать видео: {e}")
    print("Но GIF-анимация доступна для просмотра.") 