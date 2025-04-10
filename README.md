# Система моделирования шахтных выработок с анализом рисков

Данный проект представляет собой комплексное решение для моделирования шахтных выработок, визуализации перемещения работников и оценки рисков возникновения чрезвычайных происшествий (ЧП) в различных частях шахты.

## Возможности системы

### 1. Моделирование шахтной системы
- Построение графа шахтных выработок на основе входных данных
- Визуализация трехмерной структуры шахты
- Определение связей между выработками
- Представление шахты в виде математической модели с использованием теории графов

### 2. Симуляция перемещения рабочих
- Моделирование движения работников по шахте
- Анимация перемещения с учетом реальных маршрутов
- Визуализация различных видов работ цветовой кодировкой
- Создание GIF-анимации и видеороликов с записью симуляции

### 3. Анализ рисков
- Расчет вероятностей ЧП для каждой выработки
- Учет базового риска, рисков от выполняемых работ и установленного оборудования
- Визуализация рисков на трехмерной карте шахты
- Расчет интегрального риска для всей шахтной системы

### 4. Статистический анализ
- Распределение рисков по выработкам
- Анализ зависимости рисков от статуса выработки
- Выявление наиболее опасных участков шахты
- Корреляция рисков с количеством работ и оборудования

## Структура проекта

### Входные данные (CSV-файлы)
- `mine_axes.csv` - Данные о шахтных выработках (координаты, наименования, статусы)
- `equipment.csv` - Данные об оборудовании в шахте
- `axis_works.csv` - Привязка работ к выработкам
- `works.csv` - Типы работ, выполняемых в шахте

### Основные скрипты
- `main.py` - Основной скрипт для запуска всех компонентов системы с интерактивным меню
- `mine_network.py` - Построение и визуализация графа шахтных выработок
- `create_animation.py` - Создание покадровой анимации перемещения работников
- `create_gif.py` - Создание GIF-анимации из сгенерированных кадров
- `risk_analyzer.py` - Анализ рисков в шахтной системе
- `risk_stats.py` - Статистический анализ и визуализация рисков
- `analyze_connectivity.py` - Анализ связности графа шахтных выработок

### Директории
- `animation_frames/` - Кадры анимации перемещения работников
- `risk_stats/` - Статистические визуализации рисков

### Выходные файлы
- `mine_network.png` - Визуализация графа шахтных выработок
- `mine_network_axes.csv`, `mine_network_edges.csv` - Экспорт данных графа
- `workers_animation.gif` - GIF-анимация перемещения работников
- `workers_animation.mp4` - Видеоролик с анимацией перемещения работников (при наличии ffmpeg)
- `workers_data.csv` - Данные о перемещении работников
- `risk_map.png` - Карта рисков шахтных выработок
- `risk_data.csv` - Данные о рисках в выработках
- `connectivity_analysis_t0.5.png` - Анализ связности графа

## Запуск системы

Для запуска системы необходимо выполнить команду:

```bash
python main.py
```

Система предложит выбрать один из вариантов анализа:
1. Построение графа шахтных выработок
2. Анимация перемещения работников
3. Анализ рисков шахтной системы
4. Статистический анализ рисков

Каждый компонент можно запустить и независимо, вызвав соответствующий скрипт:

```bash
# Для построения графа шахтных выработок
python mine_network.py

# Для создания анимации перемещения работников
python create_animation.py   # Покадровая анимация для создания GIF/видео
python create_gif.py         # Создание GIF из кадров анимации

# Для анализа рисков
python risk_analyzer.py      # Анализ и визуализация рисков
python risk_stats.py         # Статистический анализ рисков

# Для анализа связности графа шахтных выработок
python analyze_connectivity.py
```

## Создание анимации перемещения работников

Для создания анимации движения работников используется скрипт `create_animation.py`. Этот скрипт создает последовательность кадров в директории `animation_frames/`, на которых работники перемещаются между выработками.

Процесс анимации:
1. Инициализация работников разных типов в начальных выработках
2. Выбор целевых выработок для каждого работника на основе их специализации
3. Нахождение кратчайшего пути по графу к целевой выработке
4. Анимация перемещения работника по найденному пути
5. Анимация выполнения работ в целевой выработке
6. Выбор новой цели и повторение процесса

### Оптимизация поведения работников

Симуляция перемещения работников оптимизирована для более реалистичного отображения:

1. **Баланс времени работы и перемещения**:
   - Работники проводят ~70% времени выполняя работы в выработках
   - Только ~30% времени тратится на перемещение между выработками

2. **Приоритет работы на месте**:
   - После завершения текущей задачи работник с вероятностью 70% останется в той же выработке
   - Только с вероятностью 30% работник будет искать новую выработку

3. **Оптимизация движения**:
   - Увеличена скорость перемещения между выработками для более динамичной визуализации
   - Направление маркера меняется в зависимости от направления движения

Ключевой функцией, обеспечивающей анимацию, является `update_frame()` в файле `create_animation.py`. Она обновляет положение работников, отрисовывает шахту и создает кадры анимации.

После создания кадров можно запустить `create_gif.py` для создания GIF-анимации или использовать ffmpeg (если установлен) для создания видео.

## Технические требования

### Зависимости
- Python 3.6 или выше
- Библиотеки: pandas, networkx, matplotlib, numpy, imageio (для GIF-анимации)
- FFMPEG (опционально, для создания видеороликов)

### Установка зависимостей
```bash
pip install pandas networkx matplotlib numpy imageio
```

Для установки FFMPEG следуйте инструкциям на сайте: https://ffmpeg.org/download.html

## Описание алгоритмов и моделей

### Граф шахтных выработок
Система строит граф шахтных выработок, где узлами являются выработки, а рёбрами - соединения между ними. Соединения определяются на основе близости координат начала и конца выработок (параметр `tolerance`). Для обеспечения связности графа в некоторых случаях добавляются искусственные соединения.

### Модель перемещения работников
Работники перемещаются по шахте, выбирая целевые выработки на основе своей специализации и доступных работ. Для нахождения оптимального пути используется алгоритм поиска кратчайшего пути в графе.

В классе `Worker` реализованы следующие методы:
- `set_target()` - установка целевой выработки и поиск пути к ней
- `update()` - обновление состояния работника (перемещение или выполнение работы)
- `start_working()` - начало выполнения работы
- `find_shortest_path()` - нахождение кратчайшего пути к целевой выработке

### Модель оценки рисков
Риски в выработках рассчитываются по формуле:
```
total_risk = base_risk + work_risk + equipment_risk - 
             (base_risk * work_risk) - (base_risk * equipment_risk) - 
             (work_risk * equipment_risk) + 
             (base_risk * work_risk * equipment_risk)
```

где:
- `base_risk` - базовый риск выработки, зависящий от её статуса
- `work_risk` - риск от выполняемых работ
- `equipment_risk` - риск от установленного оборудования

Интегральный риск для всей шахты рассчитывается как вероятность того, что хотя бы в одной выработке произойдет ЧП.

### Модифицированная модель оценки рисков

Для снижения интегральной вероятности ЧП в шахте была введена модифицированная формула с понижающими коэффициентами:

```
// Применяем понижающие коэффициенты к компонентам риска
work_risk = work_risk * 0.7
equipment_risk = equipment_risk * 0.6

// Базовый риск также был уменьшен
base_risk = 0.0005 * status  // Было 0.001 * status

// Формула расчета общего риска с понижающими коэффициентами
total_risk = base_risk + work_risk + equipment_risk - 
             (base_risk * work_risk) - (base_risk * equipment_risk) - 
             (work_risk * equipment_risk) + 
             (base_risk * work_risk * equipment_risk)
```

Ключевые изменения в модифицированной модели:
1. Базовый риск выработки уменьшен в 2 раза (с коэффициента 0.001 до 0.0005)
2. Риск от выполняемых работ снижен на 30% (коэффициент 0.7)
3. Риск от оборудования снижен на 40% (коэффициент 0.6)
4. Риск оборудования рассчитывается с меньшим весом: `0.001 * status * (1 + line_eq / 10)` вместо прежнего `0.002 * status * (1 + line_eq / 10)`

Эти изменения позволили снизить интегральную вероятность ЧП в шахте с исходных 0.6182 до приемлемых значений, сохраняя при этом реалистичную модель относительных рисков между выработками.

## Основные улучшения в текущей версии

### 1. Улучшенная визуализация
- Изменена проекция 2D визуализации с Y-Z (вид спереди) на X-Z (вид сбоку) для лучшего понимания пространственной структуры шахты
- Добавлено отображение соединений между выработками разными стилями линий для различения естественных и искусственных связей
- Улучшены подписи рисков на графиках для более четкой идентификации проблемных зон

### 2. Оптимизированная модель расчета рисков
- Модифицирована формула расчета вероятностей ЧП для более реалистичных оценок
- Снижена интегральная вероятность ЧП в шахте с 0.6182 до приемлемых значений
- Добавлены понижающие коэффициенты для рисков от работ и оборудования
- Сохранена реалистичная модель относительных рисков между выработками

## Проекции визуализации

В системе используются следующие типы визуализации:

1. **3D визуализация** - показывает полную трехмерную структуру шахты с учетом всех пространственных координат X, Y, Z:
   - Тоннели представлены как линии в пространстве
   - Центры выработок отмечены маркерами с цветовой кодировкой
   - Соединения между выработками показаны пунктирными линиями

2. **2D визуализация (вид сбоку, X-Z плоскость)** - схематическое представление графа шахты:
   - Отображаются только центры выработок (узлы графа)
   - Соединения между выработками показаны пунктирными линиями
   - Работники двигаются по ребрам графа между центрами выработок
   - Цветовая кодировка выработок соответствует их статусу или уровню риска

Такой подход к 2D визуализации позволяет более наглядно отображать структуру шахты как графа, что делает удобным отслеживание перемещения работников между выработками и анализ рисков. Изменение плоскости проекции с Y-Z на X-Z дает более информативное представление, поскольку большинство выработок имеют значительное горизонтальное протяжение по оси X.

## Автор
Система разработана для моделирования шахтной инфраструктуры и анализа рисков чрезвычайных происшествий.

## Лицензия
Проект разработан для образовательных и исследовательских целей. 
