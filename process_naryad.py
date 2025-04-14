import pandas as pd
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def load_naryad_data(file_path='наряд-тестовый-_new_.csv'):
    """Загружает данные наряда из CSV файла"""
    print("Пытаюсь загрузить файл наряда:", file_path)
    if os.path.exists(file_path):
        print(f"Файл найден: {file_path}")
    else:
        print(f"ОШИБКА: Файл не найден: {file_path}")
        return None
        
    try:
        naryad_df = pd.read_csv(file_path, encoding='utf-8')
        print(f"Загружен файл наряда: {file_path}")
        print(f"Количество записей: {len(naryad_df)}")
        return naryad_df
    except Exception as e:
        print(f"ОШИБКА при загрузке файла наряда: {e}")
        return None

def create_static_graph(mine_axes_file='mine_axes.csv', output_file='static_distance_graph.png'):
    """Создает статический 2D граф с весами ребер, равными расстояниям между выработками"""
    print("Пытаюсь создать статический граф")
    # Загрузка данных о выработках
    if not os.path.exists(mine_axes_file):
        print(f"ОШИБКА: Файл выработок не найден: {mine_axes_file}")
        return None
        
    try:
        mine_axes = pd.read_csv(mine_axes_file)
        print(f"Загружен файл выработок: {mine_axes_file}")
        print(f"Количество выработок: {len(mine_axes)}")
    except Exception as e:
        print(f"ОШИБКА при загрузке файла выработок: {e}")
        return None

    # Создание графа
    G = nx.Graph()
    
    # Добавление узлов (выработок)
    for idx, row in mine_axes.iterrows():
        node_id = row['short_name']
        G.add_node(node_id, 
                  name=row['full_name'],
                  status=row['status'],
                  start_pos=(row['xs'], row['ys'], row['zs']),
                  end_pos=(row['xf'], row['yf'], row['zf']),
                  length=np.sqrt((row['xf']-row['xs'])**2 + (row['yf']-row['ys'])**2 + (row['zf']-row['zs'])**2))
    
    print(f"Добавлено узлов (выработок) в граф: {len(G.nodes)}")
    
    # Поиск смежностей на основе общих координат
    tolerance = 2.0  # Расстояние, в пределах которого считаем, что выработки соединены
    
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
                ("start-start", start1, start2),
                ("start-end", start1, end2),
                ("end-start", end1, start2),
                ("end-end", end1, end2)
            ]
            
            for conn_type, point1, point2 in connections:
                distance = np.sqrt(sum((a-b)**2 for a, b in zip(point1, point2)))
                if distance < tolerance and not G.has_edge(node1, node2):
                    G.add_edge(node1, node2, weight=distance, conn_type=conn_type)
    
    print(f"Найдено естественных соединений: {len(G.edges)}")
    
    # Добавим искусственные соединения, если граф не связный
    components = list(nx.connected_components(G))
    if len(components) > 1:
        print(f"Граф не связный, найдено {len(components)} компонент. Добавляем искусственные соединения.")
        
        # Для каждой пары компонент найдем ближайшие узлы
        for i in range(len(components) - 1):
            comp1 = list(components[i])
            comp2 = list(components[i + 1])
            
            min_dist = float('inf')
            closest_pair = None
            
            for node1 in comp1:
                pos1 = G.nodes[node1]['start_pos']  # Используем начальную точку
                
                for node2 in comp2:
                    pos2 = G.nodes[node2]['start_pos']  # Используем начальную точку
                    
                    dist = np.sqrt(sum((a-b)**2 for a, b in zip(pos1, pos2)))
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (node1, node2)
            
            if closest_pair:
                G.add_edge(closest_pair[0], closest_pair[1], weight=min_dist, artificial=True)
                print(f"Добавлено искусственное соединение между {closest_pair[0]} и {closest_pair[1]}, расстояние: {min_dist:.2f}")
    else:
        print("Граф полностью связный, искусственные соединения не требуются.")
    
    # Визуализация графа с весами ребер
    print("Создание визуализации статического графа...")
    plt.figure(figsize=(14, 10))
    
    # Позиции узлов для отрисовки (используем только x и z координаты для 2D)
    pos = {}
    for node in G.nodes:
        start_pos = G.nodes[node]['start_pos']
        end_pos = G.nodes[node]['end_pos']
        pos[node] = (
            (start_pos[0] + end_pos[0]) / 2,
            (start_pos[2] + end_pos[2]) / 2  # используем z вместо y для вида сбоку
        )
    
    # Отрисовка узлов с разными цветами в зависимости от статуса
    node_colors = []
    for node in G.nodes:
        status = G.nodes[node]['status']
        if status == 1:
            node_colors.append('green')
        elif status == 2:
            node_colors.append('orange')
        elif status == 3: 
            node_colors.append('red')
        else:
            node_colors.append('blue')
    
    # Отрисовка графа
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Разделяем обычные и искусственные ребра
    regular_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('artificial', False)]
    artificial_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('artificial', True)]
    
    print(f"Обычных ребер: {len(regular_edges)}, искусственных ребер: {len(artificial_edges)}")
    
    # Отрисовка обычных ребер
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, width=1.0, alpha=0.5)
    
    # Отрисовка искусственных ребер
    nx.draw_networkx_edges(G, pos, edgelist=artificial_edges, width=1.5, edge_color='red', 
                          style='dashed', alpha=0.7)
    
    # Отрисовка весов ребер (расстояний)
    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    
    # Добавление легенды
    plt.plot([], [], 'o', color='green', label='Статус 1 - Хорошее состояние')
    plt.plot([], [], 'o', color='orange', label='Статус 2 - Требует внимания')
    plt.plot([], [], 'o', color='red', label='Статус 3 - Требует ремонта')
    plt.plot([], [], 'o', color='blue', label='Статус 4 - Другое')
    plt.plot([], [], '-', color='black', alpha=0.5, label='Естественное соединение')
    plt.plot([], [], '--', color='red', alpha=0.7, label='Искусственное соединение')
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title('Статический граф шахты с весами-расстояниями (вид сбоку, X-Z)', fontsize=14)
    plt.axis('off')
    plt.grid(False)
    plt.tight_layout()
    
    print(f"Сохраняю граф в файл: {output_file}")
    try:
        # Сохранение графа
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Статический граф сохранен в файл: {output_file}")
    except Exception as e:
        print(f"ОШИБКА при сохранении графа: {e}")
    
    return G

def process_naryad_data():
    """Обрабатывает файл наряда и интегрирует его в систему"""
    print("="*80)
    print("ОБРАБОТКА ФАЙЛА НАРЯДА И СОЗДАНИЕ СТАТИЧЕСКОГО ГРАФА")
    print("="*80)
    
    # Загрузка файла наряда
    naryad_df = load_naryad_data()
    if naryad_df is None:
        print("Не удалось загрузить файл наряда.")
        return False
        
    # Создание статического графа с весами-расстояниями
    G = create_static_graph()
    if G is None:
        print("Не удалось создать статический граф.")
        return False
    
    print("="*80)
    print("Обработка файла наряда успешно завершена.")
    print("="*80)
    return True

if __name__ == "__main__":
    process_naryad_data() 