import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Отключаем интерактивное отображение графиков
plt.ioff()  # отключает интерактивный режим - графики не будут показываться

# Чтение данных
mine_axes = pd.read_csv('mine_axes.csv')
equipment = pd.read_csv('equipment.csv')
axis_works = pd.read_csv('axis_works.csv')
works = pd.read_csv('works.csv')

# Создание графа шахтных выработок
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

# Определение смежных выработок (если конец одной выработки совпадает с началом или концом другой)
# Увеличиваем tolerance для обеспечения связности графа
tolerance = 2.0  # Увеличено с 0.5 до 2.0 для лучшей связности

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

# Информация о связности графа
print(f"Общее количество узлов (выработок): {G.number_of_nodes()}")
print(f"Общее количество ребер (соединений): {G.number_of_edges()}")

# Добавление информации об оборудовании
for idx, row in equipment.iterrows():
    equip_id = row['short_name']
    xs, ys, zs = row['xs'], row['ys'], row['zs']
    xf, yf, zf = row['xf'], row['yf'], row['zf']
    
    # Находим выработку, к которой относится оборудование
    for node in G.nodes:
        node_data = G.nodes[node]
        start_pos = node_data['start_pos']
        end_pos = node_data['end_pos']
        
        # Проверяем, лежит ли линия оборудования на линии выработки
        # (для простоты считаем, что оборудование привязано к выработке,
        # если его конечные точки очень близки к концам выработки)
        if (np.sqrt(sum((a-b)**2 for a, b in zip((xs, ys, zs), start_pos))) < tolerance and
            np.sqrt(sum((a-b)**2 for a, b in zip((xf, yf, zf), end_pos))) < tolerance):
            G.nodes[node]['equipment'].append({
                'id': equip_id,
                'name': row['full_name'],
                'status': row['status'],
                'line_eq': row['line_eq']
            })
            break

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

# Создаем класс для имитации перемещения людей
class Worker:
    def __init__(self, id, name, current_node, work_code):
        self.id = id
        self.name = name
        self.current_node = current_node
        self.work_code = work_code
        self.path_history = [current_node]
    
    def move_to(self, next_node):
        """Перемещает работника в новую выработку"""
        if G.has_edge(self.current_node, next_node):
            self.current_node = next_node
            self.path_history.append(next_node)
            return True
        return False
    
    def find_shortest_path(self, target_node):
        """Находит кратчайший путь до целевой выработки"""
        try:
            path = nx.shortest_path(G, self.current_node, target_node)
            return path
        except nx.NetworkXNoPath:
            print(f"Не удалось найти путь от {self.current_node} до {target_node}!")
            # Поскольку граф связный, этого не должно происходить
            return None

# Визуализация графа
def visualize_mine_graph():
    # Создаем фигуру с двумя подграфиками - 3D и 2D
    fig = plt.figure(figsize=(20, 10))
    
    # 3D-визуализация
    ax1 = fig.add_subplot(121, projection='3d')
    
    # 2D-визуализация (вид сбоку: x-z плоскость)
    ax2 = fig.add_subplot(122)
    
    # Позиции узлов будут соответствовать средней точке между началом и концом выработки
    pos_3d = {}
    for node in G.nodes:
        start_pos = G.nodes[node]['start_pos']
        end_pos = G.nodes[node]['end_pos']
        pos_3d[node] = (
            (start_pos[0] + end_pos[0]) / 2,
            (start_pos[1] + end_pos[1]) / 2,
            (start_pos[2] + end_pos[2]) / 2
        )
    
    # Отрисовка узлов в 3D
    for node in G.nodes:
        x, y, z = pos_3d[node]
        
        # Цвет узла зависит от статуса выработки
        status = G.nodes[node]['status']
        if status == 1:
            color = 'green'
        elif status == 2:
            color = 'yellow'
        elif status == 3:
            color = 'orange'
        else:
            color = 'red'
            
        ax1.scatter(x, y, z, color=color, s=100)
        ax1.text(x, y, z, node, fontsize=10)
        
        # Отрисовка линий выработок в 3D
        start_x, start_y, start_z = G.nodes[node]['start_pos']
        end_x, end_y, end_z = G.nodes[node]['end_pos']
        ax1.plot([start_x, end_x], [start_y, end_y], [start_z, end_z], 'k-', linewidth=1)
    
    # Отрисовка ребер (связей между выработками) в 3D
    for edge in G.edges:
        node1, node2 = edge
        x1, y1, z1 = pos_3d[node1]
        x2, y2, z2 = pos_3d[node2]
        
        # Проверяем, является ли это искусственным соединением
        if G.edges[edge].get('artificial', False):
            # Искусственные соединения рисуем красным пунктиром
            ax1.plot([x1, x2], [y1, y2], [z1, z2], 'r--', linewidth=1.5, alpha=0.7)
        else:
            # Обычные соединения рисуем синим пунктиром
            ax1.plot([x1, x2], [y1, y2], [z1, z2], 'b--', linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Модель шахтных выработок')
    
    # Отрисовка 2D-проекции (вид сбоку: x-z плоскость)
    for node in G.nodes:
        # Цвет узла зависит от статуса выработки
        status = G.nodes[node]['status']
        if status == 1:
            color = 'green'
        elif status == 2:
            color = 'yellow'
        elif status == 3:
            color = 'orange'
        else:
            color = 'red'
            
        # Отрисовка линий выработок в 2D (x-z проекция)
        start_x, start_z = G.nodes[node]['start_pos'][0], G.nodes[node]['start_pos'][2]
        end_x, end_z = G.nodes[node]['end_pos'][0], G.nodes[node]['end_pos'][2]
        ax2.plot([start_x, end_x], [start_z, end_z], 'k-', linewidth=1)
        
        # Рисуем узел в середине выработки
        middle_x = (start_x + end_x) / 2
        middle_z = (start_z + end_z) / 2
        ax2.scatter(middle_x, middle_z, color=color, s=100)
        ax2.text(middle_x, middle_z, node, fontsize=10)
    
    # Отрисовка ребер (связей между выработками) в 2D
    for edge in G.edges:
        node1, node2 = edge
        x1, z1 = pos_3d[node1][0], pos_3d[node1][2]
        x2, z2 = pos_3d[node2][0], pos_3d[node2][2]
        
        # Проверяем, является ли это искусственным соединением
        if G.edges[edge].get('artificial', False):
            # Искусственные соединения рисуем красным пунктиром
            ax2.plot([x1, x2], [z1, z2], 'r--', linewidth=1.5, alpha=0.7)
        else:
            # Обычные соединения рисуем синим пунктиром
            ax2.plot([x1, x2], [z1, z2], 'b--', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title('2D Модель шахтных выработок (вид сбоку)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('mine_network.png', dpi=300)
    plt.show()

# Пример создания работников и их перемещения
def simulate_workers_movement():
    workers = []
    
    # Создаем нескольких работников
    for i in range(5):
        # Выбираем случайную выработку для начального размещения
        start_node = list(G.nodes)[i]
        
        # Если в выработке есть связанные работы, выбираем одну из них
        if G.nodes[start_node]['workers']:
            work = G.nodes[start_node]['workers'][0]
            worker = Worker(
                id=f"W{i+1}",
                name=f"Работник {i+1}",
                current_node=start_node,
                work_code=work['work_code']
            )
            workers.append(worker)
    
    # Моделируем перемещение работников
    for worker in workers:
        print(f"{worker.name} начинает в выработке {worker.current_node}")
        
        # Выбираем случайную целевую выработку
        target_node = list(G.nodes)[np.random.randint(0, len(G.nodes))]
        while target_node == worker.current_node:
            target_node = list(G.nodes)[np.random.randint(0, len(G.nodes))]
        
        # Находим и выполняем путь
        path = worker.find_shortest_path(target_node)
        if path:
            print(f"Найден путь до {target_node}: {path}")
            for next_node in path[1:]:
                success = worker.move_to(next_node)
                if success:
                    print(f"  Перемещение в {next_node}")
        else:
            print(f"Путь до {target_node} не найден")
    
    return workers

# Сохранение обновленных данных в CSV
def save_to_csv():
    # Подготовка данных для выработок
    axes_data = []
    for node in G.nodes:
        node_data = G.nodes[node]
        
        # Добавляем оборудование и работников в виде строк
        equipment_str = ', '.join([eq['id'] for eq in node_data['equipment']])
        workers_str = ', '.join([str(w['work_code']) for w in node_data['workers']])
        
        # Список смежных выработок
        adjacent_nodes = list(G.neighbors(node))
        adjacent_str = ', '.join(adjacent_nodes)
        
        axes_data.append({
            'short_name': node,
            'full_name': node_data['name'],
            'status': node_data['status'],
            'act_works': node_data['act_works'],
            'xs': node_data['start_pos'][0],
            'ys': node_data['start_pos'][1],
            'zs': node_data['start_pos'][2],
            'xf': node_data['end_pos'][0],
            'yf': node_data['end_pos'][1],
            'zf': node_data['end_pos'][2],
            'equipment': equipment_str,
            'works': workers_str,
            'adjacent_axes': adjacent_str,
            'length': node_data['length']
        })
    
    # Создаем DataFrame и сохраняем в CSV
    axes_df = pd.DataFrame(axes_data)
    axes_df.to_csv('mine_network_axes.csv', index=False)
    
    # Сохраняем также информацию о смежностях (ребрах графа)
    edges_data = []
    for edge in G.edges:
        node1, node2 = edge
        edge_data = G.edges[edge]
        edges_data.append({
            'Node1': node1,
            'Node2': node2,
            'Weight': edge_data['weight'],
            'Artificial': edge_data.get('artificial', False)
        })
    edges_df = pd.DataFrame(edges_data)
    edges_df.to_csv('mine_network_edges.csv', index=False)

if __name__ == "__main__":
    # Визуализируем граф
    visualize_mine_graph()
    
    # Запускаем симуляцию перемещения работников
    workers = simulate_workers_movement()
    
    # Сохраняем результаты в CSV
    save_to_csv()
    
    print("Анализ шахтной сети завершен. Результаты сохранены в CSV файлы.") 