import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random
import time
import os
import subprocess
import sys
import matplotlib.animation as animation

print("Начинаем создание анимации перемещения работников...")

# Чтение данных
print("Загрузка данных из CSV файлов...")
mine_axes = pd.read_csv('mine_axes.csv')
equipment = pd.read_csv('equipment.csv')
axis_works = pd.read_csv('axis_works.csv')
works = pd.read_csv('works.csv')

# Словарь соответствия кодов работ и их названий для быстрого доступа
work_names = {}
for _, row in works.iterrows():
    work_names[row['work_code']] = row['full_name']

# Словарь с цветами для разных типов работ
work_type_colors = {
    100: 'blue',     # Работы в лаве
    200: 'green',    # Проходческие работы
    300: 'purple',   # Транспортные работы
    400: 'orange',   # Работы по вентиляции
    500: 'cyan',     # Ремонтные работы
}

# Создание графа шахтных выработок
print("Создание графа шахтных выработок...")
G = nx.Graph()

# Добавление узлов (выработок)
for idx, row in mine_axes.iterrows():
    node_id = row['short_name']
    G.add_node(node_id, 
               name=row['full_name'],
               status=row['status'],
               act_works=row['act_works'],
               start_pos=(row['xs'], row['ys'], row['zs']),
               end_pos=(row['xf'], row['yf'], row['zf']),
               length=np.sqrt((row['xf']-row['xs'])**2 + (row['yf']-row['ys'])**2 + (row['zf']-row['zs'])**2),
               equipment=[],
               workers=[])

# Определение смежных выработок
tolerance = 2.0  # Используем то же значение, что и в mine_network.py

# Поиск смежностей на основе общих координат
for i, row1 in mine_axes.iterrows():
    node1 = row1['short_name']
    start1 = (row1['xs'], row1['ys'], row1['zs'])
    end1 = (row1['xf'], row1['yf'], row1['zf'])
    
    for j, row2 in mine_axes.iterrows():
        if i == j:  # Пропускаем ту же самую выработку
            continue
        
        node2 = row2['short_name']
        start2 = (row2['xs'], row2['ys'], row2['zs'])
        end2 = (row2['xf'], row2['yf'], row2['zf'])
        
        # Проверяем все возможные соединения между концами отрезков
        connections = [
            (start1, start2),
            (start1, end2),
            (end1, start2),
            (end1, end2)
        ]
        
        for point1, point2 in connections:
            distance = np.sqrt(sum((a-b)**2 for a, b in zip(point1, point2)))
            if distance < tolerance and not G.has_edge(node1, node2):
                G.add_edge(node1, node2, weight=distance)

# Проверка связности графа
if not nx.is_connected(G):
    print("ВНИМАНИЕ: Граф шахтных выработок не является связным!")
    print("Добавляем искусственные соединения для обеспечения связности.")
    
    # Находим компоненты связности
    components = list(nx.connected_components(G))
    
    if len(components) > 1:
        # Соединяем компоненты, добавляя ребра между ближайшими узлами
        for i in range(len(components) - 1):
            comp1 = components[i]
            comp2 = components[i + 1]
            
            # Находим ближайшие узлы из разных компонент
            min_distance = float('inf')
            closest_nodes = None
            
            for node1 in comp1:
                pos1 = G.nodes[node1]['start_pos']  # Используем начало выработки
                
                for node2 in comp2:
                    pos2 = G.nodes[node2]['start_pos']
                    
                    distance = np.sqrt(sum((a-b)**2 for a, b in zip(pos1, pos2)))
                    if distance < min_distance:
                        min_distance = distance
                        closest_nodes = (node1, node2)
            
            # Добавляем ребро между ближайшими узлами
            if closest_nodes:
                node1, node2 = closest_nodes
                G.add_edge(node1, node2, weight=min_distance, artificial=True)
                print(f"Добавлено искусственное соединение между {node1} и {node2} с расстоянием {min_distance:.2f}")
else:
    print("Граф шахтных выработок связен! Между любыми двумя выработками существует путь.")

# Связывание работ с выработками
work_mapping = {}
for _, row in axis_works.iterrows():
    axis = row['short_name']
    work_code = row['work_code']
    
    if axis not in work_mapping:
        work_mapping[axis] = []
    
    work_mapping[axis].append(work_code)

# Добавляем работы к узлам графа
for node in G.nodes:
    if node in work_mapping:
        work_codes = work_mapping[node]
        workers_info = []
        
        for code in work_codes:
            work_data = works[works['work_code'] == code]
            if not work_data.empty:
                workers_info.append({
                    'work_code': code,
                    'name': work_data.iloc[0]['full_name'],
                    'risk': work_data.iloc[0]['ud_risk'],
                    'color': work_data.iloc[0]['col_work']
                })
        
        G.nodes[node]['workers'] = workers_info

# Позиции узлов для отрисовки графа
pos_3d = {}
for node in G.nodes:
    start_pos = G.nodes[node]['start_pos']
    end_pos = G.nodes[node]['end_pos']
    pos_3d[node] = (
        (start_pos[0] + end_pos[0]) / 2,
        (start_pos[1] + end_pos[1]) / 2,
        (start_pos[2] + end_pos[2]) / 2
    )

# Класс работника
class Worker:
    def __init__(self, id, name, current_node, work_code):
        self.id = id
        self.name = name
        self.current_node = current_node
        self.work_code = work_code
        self.path_history = [current_node]
        self.target_node = None
        self.path = [current_node]  # Инициализируем начальным узлом вместо None
        self.current_path_index = 0
        
        # Текущая выработка и позиция на ней
        node_data = G.nodes[current_node]
        self.current_segment = (node_data['start_pos'], node_data['end_pos'])
        self.segment_progress = random.random()  # Начальная случайная позиция на выработке
        
        # Вычисляем начальную 3D позицию на выработке
        start_pos = node_data['start_pos']
        end_pos = node_data['end_pos']
        self.position = (
            start_pos[0] + (end_pos[0] - start_pos[0]) * self.segment_progress,
            start_pos[1] + (end_pos[1] - start_pos[1]) * self.segment_progress,
            start_pos[2] + (end_pos[2] - start_pos[2]) * self.segment_progress
        )
        
        self.next_position = None
        self.movement_progress = 0
        
        # Определяем цвет работника по типу работы
        work_type = (self.work_code // 100) * 100
        self.color = work_type_colors.get(work_type, 'magenta')
        
        self.speed = random.uniform(0.4, 0.6)  # Увеличиваем скорость перемещения (было 0.15-0.3)
        self.work_time = random.randint(20, 35)  # Время работы
        self.current_work_time = 0
        self.working = False
        self.status = "ожидание"
        self.shape = 'o'  # Форма маркера
        self.size = 100    # Размер маркера
        
        # Для визуализации активности
        self.active = False
        self.active_counter = 0
        
        # Информация о выполняемой задаче
        self.task = {
            'description': f"Выполняет: {work_names.get(self.work_code, 'Неизвестная работа')}",
            'progress': 0,
            'total': 100,
            'location': current_node
        }
    
    def set_target(self, target_node):
        """Устанавливает целевую выработку и находит путь к ней"""
        self.target_node = target_node
        found_path = self.find_shortest_path(target_node)
        
        if found_path and len(found_path) > 1:
            self.path = found_path
            self.current_path_index = 0
            
            # Устанавливаем первый сегмент пути - текущую выработку
            node_data = G.nodes[self.path[0]]
            self.current_segment = (node_data['start_pos'], node_data['end_pos'])
            self.status = "перемещение"
            self.working = False
            self.shape = '>'  # Изменяем форму маркера при движении
            return True
        else:
            # Если путь не найден или состоит только из текущего узла, 
            # начинаем работать на месте
            self.start_working()
            return False
    
    def start_working(self):
        """Начинает выполнять работу в текущей выработке"""
        self.working = True
        self.current_work_time = 0
        self.work_time = random.randint(20, 35)  # Значительно увеличиваем время работы (было 3-10)
        self.status = "работа"
        self.shape = '*'  # Звездочка для работающего состояния
        self.size = 150   # Увеличиваем размер при работе
        
        # Обновляем информацию о задаче
        self.task = {
            'description': f"Выполняет: {work_names.get(self.work_code, 'Неизвестная работа')}",
            'progress': 0,
            'total': self.work_time,
            'location': self.current_node
        }
    
    def update(self):
        """Обновляет состояние работника"""
        # Если работник выполняет работу
        if self.working:
            self.current_work_time += 1  # Уменьшаем скорость работы (было 3)
            self.task['progress'] = self.current_work_time
            
            # Пульсация при работе
            self.active_counter = (self.active_counter + 1) % 10
            self.active = self.active_counter < 5
            
            # Если работа завершена, ищем новую цель
            if self.current_work_time >= self.work_time:
                self.working = False
                self.status = "ожидание"
                self.shape = 'o'
                self.size = 100
                
                # С большой вероятностью продолжаем работать в текущей выработке
                if random.random() < 0.7:  # Увеличили вероятность остаться на месте (было 0.1)
                    self.start_working()
                    return True
                
                # Выбираем случайную выработку, исключая текущую
                nodes = list(G.nodes)
                if self.current_node in nodes:
                    nodes.remove(self.current_node)
                if nodes:
                    target = random.choice(nodes)
                    self.set_target(target)
                    return True
                
                # Выбор новой цели с приоритетом выработок с подходящими работами
                suitable_nodes = []
                for node in G.nodes:
                    if node != self.current_node:
                        # Проверяем доступность этого типа работы в выработке
                        node_data = G.nodes[node]
                        for work_info in node_data.get('workers', []):
                            if work_info['work_code'] == self.work_code:
                                suitable_nodes.append(node)
                                break
                
                if suitable_nodes:
                    # Выбираем случайную выработку из подходящих
                    target = random.choice(suitable_nodes)
                else:
                    # Если нет подходящих, выбираем любую случайную выработку
                    target = random.choice(list(G.nodes))
                    while target == self.current_node:
                        target = random.choice(list(G.nodes))
                
                self.set_target(target)
                return True
            return False
            
        # Если работник перемещается и у него есть путь
        elif self.path and self.current_path_index < len(self.path):
            current_node = self.path[self.current_path_index]
            node_data = G.nodes[current_node]
            
            # Получаем координаты текущей выработки
            start_pos = node_data['start_pos']
            end_pos = node_data['end_pos']
            
            # Обновляем позицию на текущем сегменте выработки
            self.segment_progress += self.speed * 0.3  # Увеличиваем скорость движения вдоль выработки (было 0.1)
            
            if self.segment_progress >= 1.0:
                # Достигли конца текущей выработки
                self.segment_progress = 0.0  # Сбрасываем прогресс
                
                # Переходим к следующей выработке в пути, если есть
                if self.current_path_index < len(self.path) - 1:
                    self.current_path_index += 1
                    self.current_node = self.path[self.current_path_index]
                    self.path_history.append(self.current_node)
                    
                    # Получаем данные следующей выработки
                    next_node_data = G.nodes[self.current_node]
                    self.current_segment = (next_node_data['start_pos'], next_node_data['end_pos'])
                else:
                    # Достигли последней выработки в пути - начинаем работать
                    self.start_working()
                    return True
            
            # Обновляем текущую позицию в выработке
            self.position = (
                start_pos[0] + (end_pos[0] - start_pos[0]) * self.segment_progress,
                start_pos[1] + (end_pos[1] - start_pos[1]) * self.segment_progress,
                start_pos[2] + (end_pos[2] - start_pos[2]) * self.segment_progress
            )
            
            # Определяем направление движения для маркера
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            dz = end_pos[2] - start_pos[2]
            
            # Определяем направление маркера в зависимости от направления движения
            if abs(dx) > abs(dy) and abs(dx) > abs(dz):
                self.shape = '>' if dx > 0 else '<'
            elif abs(dy) > abs(dx) and abs(dy) > abs(dz):
                self.shape = '^' if dy > 0 else 'v'
            else:
                self.shape = 'd' if dz > 0 else 's'
            
            return True
        else:
            # Не перемещается и не работает - начинаем работать
            self.start_working()
            return True
    
    def find_shortest_path(self, target_node):
        """Находит кратчайший путь до целевой выработки"""
        try:
            path = nx.shortest_path(G, self.current_node, target_node)
            return path
        except nx.NetworkXNoPath:
            print(f"Не удалось найти путь от {self.current_node} до {target_node}!")
            # Возвращаем список с текущим узлом вместо None
            return [self.current_node]

# Подготовка анимации
print("Подготовка анимации...")

# Создаем директорию для кадров, если её нет
if not os.path.exists('animation_frames'):
    os.makedirs('animation_frames')

# Создаем работников
print("Создание работников...")
work_types = [
    (101, "Шахтер"), (102, "Крепильщик"), (201, "Бурильщик"), 
    (203, "Взрывник"), (301, "Оператор конвейера"), (304, "Машинист"),
    (401, "Вентиляционщик"), (501, "Ремонтник"), (503, "Электрик")
]

workers = []
for code, name in work_types:
    # Для каждого типа создаем несколько работников
    num_workers = random.randint(1, 3)
    for i in range(num_workers):
        # Находим подходящие выработки для этого типа работы
        suitable_nodes = []
        for node in G.nodes:
            work_codes = [w['work_code'] for w in G.nodes[node].get('workers', [])]
            if code in work_codes:
                suitable_nodes.append(node)
        
        # Если нет подходящих, используем любую
        if not suitable_nodes:
            suitable_nodes = list(G.nodes)
        
        # Выбираем случайную выработку
        start_node = random.choice(suitable_nodes)
        
        # Создаем работника
        worker = Worker(
            id=f"{name}-{i+1}",
            name=f"{name} {i+1}",
            current_node=start_node,
            work_code=code
        )
        
        # Сразу заставляем его начать работать
        worker.start_working()
        workers.append(worker)

print(f"Создано {len(workers)} работников")

# Определяем границы графика
min_x = min([G.nodes[node]['start_pos'][0] for node in G.nodes] + [G.nodes[node]['end_pos'][0] for node in G.nodes])
max_x = max([G.nodes[node]['start_pos'][0] for node in G.nodes] + [G.nodes[node]['end_pos'][0] for node in G.nodes])
min_y = min([G.nodes[node]['start_pos'][1] for node in G.nodes] + [G.nodes[node]['end_pos'][1] for node in G.nodes])
max_y = max([G.nodes[node]['start_pos'][1] for node in G.nodes] + [G.nodes[node]['end_pos'][1] for node in G.nodes])
min_z = min([G.nodes[node]['start_pos'][2] for node in G.nodes] + [G.nodes[node]['end_pos'][2] for node in G.nodes])
max_z = max([G.nodes[node]['start_pos'][2] for node in G.nodes] + [G.nodes[node]['end_pos'][2] for node in G.nodes])

# Создаем фигуру для визуализации
fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(121, projection='3d')  # 3D-визуализация слева
ax2 = fig.add_subplot(122)  # 2D-визуализация справа (вид спереди)

# Устанавливаем количество кадров анимации
num_frames = 100
print(f"Будет создано {num_frames} кадров анимации")

# Функция обновления кадра анимации
def update_frame(frame_num, workers, ax1, ax2):
    ax1.clear()
    ax2.clear()
    
    # Отрисовка выработок (линий) в 3D
    for node, data in G.nodes(data=True):
        start_pos = data['start_pos']
        end_pos = data['end_pos']
        ax1.plot([start_pos[0], end_pos[0]], 
                [start_pos[1], end_pos[1]], 
                [start_pos[2], end_pos[2]], 
                'gray', alpha=0.5, linewidth=1)
    
    # В 2D не отображаем тоннели, только центры выработок
    
    # Отрисовка узлов в 3D
    for node in G.nodes:
        node_data = G.nodes[node]
        x, y, z = pos_3d[node]
        
        if node_data['status'] == 1:  # Хорошее состояние
            color = 'green'
        elif node_data['status'] == 2:  # Требует внимания
            color = 'orange'
        elif node_data['status'] == 3:  # Требует ремонта
            color = 'red'
        else:
            color = 'blue'
            
        ax1.scatter([x], [y], [z], color=color, s=20, alpha=0.7)
        
        # Отрисовка узлов в 2D (вид сбоку: x-z плоскость)
        ax2.scatter([x], [z], color=color, s=20, alpha=0.7)
        ax2.annotate(node, (x, z), fontsize=8)
        
    # Отрисовка ребер в 3D
    for u, v, data in G.edges(data=True):
        x1, y1, z1 = pos_3d[u]
        x2, y2, z2 = pos_3d[v]
        
        # Если ребро искусственное, рисуем пунктиром
        if data.get('artificial', False):
            ax1.plot([x1, x2], [y1, y2], [z1, z2], 'red', linestyle='--', alpha=0.7, linewidth=0.8)
        else:
            ax1.plot([x1, x2], [y1, y2], [z1, z2], 'black', alpha=0.3, linewidth=0.8)
            
        # Отрисовка ребер в 2D (вид сбоку: x-z плоскость)
        if data.get('artificial', False):
            ax2.plot([x1, x2], [z1, z2], 'red', linestyle='--', alpha=0.7, linewidth=0.8)
        else:
            ax2.plot([x1, x2], [z1, z2], 'blue', linestyle='--', alpha=0.5, linewidth=0.8)
    
    # Перемещение и отрисовка работников
    for worker in workers:
        worker.update()  # Обновляем состояние работника
        x, y, z = worker.position
        
        # Если работник активен (работает), менять его размер
        marker_size = worker.size * 1.2 if worker.working and worker.active else worker.size
        
        # Отрисовка в 3D (по-прежнему внутри выработок)
        ax1.scatter([x], [y], [z], color=worker.color, s=marker_size, marker=worker.shape, 
                  label=f"{worker.name} ({worker.id})" if frame_num == 0 else "")
        
        # Для 2D отрисовки используем другую логику - работники двигаются по ребрам графа
        # Вычисляем 2D позицию на ребре между центрами выработок
        if worker.path and worker.current_path_index < len(worker.path) - 1:
            # Если работник движется между двумя выработками
            current_node = worker.path[worker.current_path_index]
            next_node = worker.path[worker.current_path_index + 1]
            
            # Центр текущей выработки
            current_node_data = G.nodes[current_node]
            current_x = (current_node_data['start_pos'][0] + current_node_data['end_pos'][0]) / 2
            current_z = (current_node_data['start_pos'][2] + current_node_data['end_pos'][2]) / 2
            
            # Центр следующей выработки
            next_node_data = G.nodes[next_node]
            next_x = (next_node_data['start_pos'][0] + next_node_data['end_pos'][0]) / 2
            next_z = (next_node_data['start_pos'][2] + next_node_data['end_pos'][2]) / 2
            
            # Интерполируем положение на ребре
            edge_progress = worker.segment_progress  # Используем тот же прогресс
            x_2d = current_x + (next_x - current_x) * edge_progress
            z_2d = current_z + (next_z - current_z) * edge_progress
        else:
            # Если работник находится в выработке (не движется или работает)
            current_node = worker.current_node
            current_node_data = G.nodes[current_node]
            x_2d = (current_node_data['start_pos'][0] + current_node_data['end_pos'][0]) / 2
            z_2d = (current_node_data['start_pos'][2] + current_node_data['end_pos'][2]) / 2
                  
        # Отрисовка в 2D (вид сбоку: x-z плоскость) по ребрам графа
        ax2.scatter([x_2d], [z_2d], color=worker.color, s=marker_size, marker=worker.shape, 
                  label=f"{worker.name} ({worker.id})" if frame_num == 0 else "")
    
    # Настройка 3D вида
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Настройка 2D вида (вид сбоку: x-z плоскость)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title('Граф шахты - схематический вид сбоку (X-Z)')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Ограничение осей в 3D
    ax1.set_xlim(min_x - 10, max_x + 10)
    ax1.set_ylim(min_y - 10, max_y + 10)
    ax1.set_zlim(min_z - 10, max_z + 10)
    
    # Ограничение осей в 2D (вид сбоку: x-z плоскость)
    ax2.set_xlim(min_x - 10, max_x + 10)
    ax2.set_ylim(min_z - 10, max_z + 10)
    
    # Добавление информации о текущем кадре
    frame_time = frame_num * 5  # Условное время (5 секунд на кадр)
    minutes = frame_time // 60
    seconds = frame_time % 60
    ax1.set_title(f'Перемещение работников в шахте [Время: {minutes:02d}:{seconds:02d}, Кадр: {frame_num+1}/{num_frames}]')
    
    # Отображаем информацию о некоторых работниках
    visible_workers = sorted(workers, key=lambda w: 0 if w.working else 1)[:5]
    
    for i, worker in enumerate(visible_workers):
        progress = ""
        if worker.working:
            progress_pct = int((worker.current_work_time / worker.work_time) * 100)
            progress = f" | Прогресс: {progress_pct}%"
        
        # Определяем более подробный статус
        status_text = worker.status
        if worker.status == "работа":
            status_text = f"работает: {work_names.get(worker.work_code, 'Неизвестная работа')}"
        elif worker.status == "перемещение":
            if worker.target_node:
                status_text = f"идет в {worker.target_node}"
                
        description = f"{worker.name} ({worker.id}): {status_text}{progress}"
        
        # Определяем цвет фона для текста в зависимости от статуса
        bbox_color = 'white'
        if worker.working:
            bbox_color = 'lightgreen'
        elif worker.status == "перемещение":
            bbox_color = 'lightyellow'
        
        ax1.text2D(0.02, 0.95 - i*0.05, description, 
                 transform=ax1.transAxes,
                 bbox=dict(facecolor=bbox_color, alpha=0.7, boxstyle='round'))
    
    return ax1, ax2

# Создание кадров анимации
print("Создание кадров анимации...")
for frame in range(num_frames):
    update_frame(frame, workers, ax1, ax2)
    plt.savefig(f'animation_frames/frame_{frame:04d}.png', dpi=100, bbox_inches='tight')
    print(f'\rСоздан кадр {frame+1}/{num_frames}', end='')

print("\nКадры успешно созданы.")

# Сохранение данных о работниках
print("Сохранение данных о работниках...")
worker_data = []
for worker in workers:
    worker_data.append({
        'id': worker.id,
        'name': worker.name,
        'work_code': worker.work_code,
        'work_type': work_names.get(worker.work_code, 'Неизвестная работа'),
        'path_history': ','.join(worker.path_history),
        'num_nodes_visited': len(set(worker.path_history))
    })

worker_df = pd.DataFrame(worker_data)
worker_df.to_csv('workers_data.csv', index=False)
print("Данные о работниках сохранены в workers_data.csv")

# Создание видео из кадров с помощью ffmpeg
print("Создание видео из кадров с помощью ffmpeg...")
try:
    subprocess.run(['ffmpeg', '-framerate', '10', '-i', 'animation_frames/frame_%04d.png', 
                   '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 'workers_animation.mp4'])
    print("Видео успешно создано: workers_animation.mp4")
except Exception as e:
    print(f"ОШИБКА при создании видео: {e}")

print("Процесс завершен.") 