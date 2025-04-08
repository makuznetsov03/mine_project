import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def analyze_graph_connectivity(tolerance=0.5, visualize=True):
    """Анализирует связность графа шахтных выработок с разными параметрами tolerance"""
    print(f"Анализ связности графа с tolerance={tolerance}")
    
    # Чтение данных
    mine_axes = pd.read_csv('mine_axes.csv')
    
    # Создание графа шахтных выработок
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
    
    # Поиск смежностей на основе общих координат
    num_connections = 0
    
    # Список для хранения информации о соединениях
    connection_info = []
    
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
                    num_connections += 1
                    connection_info.append({
                        "node1": node1, 
                        "node2": node2, 
                        "conn_type": conn_type,
                        "distance": distance
                    })
    
    # Анализ компонент связности
    connected_components = list(nx.connected_components(G))
    
    print(f"Общее количество узлов (выработок): {G.number_of_nodes()}")
    print(f"Общее количество ребер (связей): {G.number_of_edges()}")
    print(f"Количество компонент связности: {len(connected_components)}")
    
    print("\nКомпоненты связности:")
    for i, component in enumerate(connected_components):
        print(f"  Компонента {i+1}: {', '.join(sorted(component))} ({len(component)} узлов)")
    
    # Проверка, является ли граф связным
    if nx.is_connected(G):
        print("\nГраф ПОЛНОСТЬЮ СВЯЗЕН! Между любыми двумя выработками существует путь.")
    else:
        print("\nГраф НЕ СВЯЗЕН! Существуют выработки, между которыми нет пути.")
        
        # Находим пары узлов, между которыми нет пути
        isolated_pairs = []
        nodes = list(G.nodes())
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                node1, node2 = nodes[i], nodes[j]
                if not nx.has_path(G, node1, node2):
                    isolated_pairs.append((node1, node2))
        
        if len(isolated_pairs) > 20:
            print(f"Найдено {len(isolated_pairs)} пар узлов без пути между ними (выводим первые 20):")
            for node1, node2 in isolated_pairs[:20]:
                print(f"  Нет пути между {node1} и {node2}")
        else:
            print(f"Найдено {len(isolated_pairs)} пар узлов без пути между ними:")
            for node1, node2 in isolated_pairs:
                print(f"  Нет пути между {node1} и {node2}")
    
    # Анализ найденных соединений
    print("\nАнализ найденных соединений:")
    conn_df = pd.DataFrame(connection_info)
    if not conn_df.empty:
        conn_type_counts = conn_df['conn_type'].value_counts()
        print(f"Типы соединений:")
        for conn_type, count in conn_type_counts.items():
            print(f"  {conn_type}: {count}")
        
        print(f"\nСтатистика расстояний между соединенными узлами:")
        print(f"  Минимальное: {conn_df['distance'].min():.4f}")
        print(f"  Максимальное: {conn_df['distance'].max():.4f}")
        print(f"  Среднее: {conn_df['distance'].mean():.4f}")
    else:
        print("  Соединения не найдены!")
    
    # Визуализация графа в 3D
    if visualize:
        visualize_graph(G, mine_axes, tolerance)
    
    return G, connected_components

def visualize_graph(G, mine_axes, tolerance):
    """Визуализирует граф шахтных выработок в 3D"""
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Используем различные цвета для разных компонент связности
    components = list(nx.connected_components(G))
    colors = plt.cm.tab10(np.linspace(0, 1, len(components)))
    
    # Позиции узлов для отрисовки
    pos_3d = {}
    for node in G.nodes:
        row = mine_axes[mine_axes['short_name'] == node].iloc[0]
        start_pos = (row['xs'], row['ys'], row['zs'])
        end_pos = (row['xf'], row['yf'], row['zf'])
        
        # Середина отрезка
        pos_3d[node] = (
            (start_pos[0] + end_pos[0]) / 2,
            (start_pos[1] + end_pos[1]) / 2,
            (start_pos[2] + end_pos[2]) / 2
        )
    
    # Отображаем выработки и их связи, используя разные цвета для компонент
    for i, component in enumerate(components):
        color = colors[i]
        
        # Отрисовка узлов компоненты
        for node in component:
            x, y, z = pos_3d[node]
            ax.scatter(x, y, z, color=color, s=100, alpha=0.8)
            ax.text(x, y, z, node, fontsize=8, color='black')
            
            # Отрисовка выработки как линии
            row = mine_axes[mine_axes['short_name'] == node].iloc[0]
            start_x, start_y, start_z = row['xs'], row['ys'], row['zs']
            end_x, end_y, end_z = row['xf'], row['yf'], row['zf']
            ax.plot([start_x, end_x], [start_y, end_y], [start_z, end_z], 'k-', linewidth=1, alpha=0.3)
        
        # Отрисовка ребер внутри компоненты
        for node1 in component:
            for node2 in G.neighbors(node1):
                if node2 in component and node1 < node2:  # Избегаем дублирования
                    x1, y1, z1 = pos_3d[node1]
                    x2, y2, z2 = pos_3d[node2]
                    ax.plot([x1, x2], [y1, y2], [z1, z2], '--', color=color, linewidth=1.5, alpha=0.6)
    
    # Настройка осей
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D визуализация графа шахтных выработок (tolerance={tolerance})')
    
    plt.tight_layout()
    plt.savefig(f'connectivity_analysis_t{tolerance}.png', dpi=200)
    plt.show()

def identify_missing_connections(tolerance_range=[0.5, 1.0, 2.0, 5.0]):
    """Анализирует, как изменяется связность при разных значениях tolerance"""
    results = {}
    
    for tolerance in tolerance_range:
        G, components = analyze_graph_connectivity(tolerance, visualize=(tolerance == tolerance_range[0]))
        
        results[tolerance] = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'num_components': len(components),
            'components': components
        }
    
    # Анализ изменений при увеличении tolerance
    print("\n\nАнализ изменений при увеличении tolerance:")
    
    prev_tolerance = None
    for tolerance in tolerance_range:
        if prev_tolerance is not None:
            new_edges = results[tolerance]['num_edges'] - results[prev_tolerance]['num_edges']
            new_components = results[prev_tolerance]['num_components'] - results[tolerance]['num_components']
            
            print(f"\nПри увеличении tolerance с {prev_tolerance} до {tolerance}:")
            print(f"  Добавлено новых связей: {new_edges}")
            print(f"  Уменьшение числа компонент: {new_components}")
            
            if new_components > 0:
                print("  Некоторые компоненты соединились!")
        
        prev_tolerance = tolerance
    
    # Рекомендация оптимального значения tolerance
    optimal_tolerance = None
    for tolerance in tolerance_range:
        if results[tolerance]['num_components'] == 1:
            optimal_tolerance = tolerance
            break
    
    if optimal_tolerance is not None:
        print(f"\nРекомендуемое значение tolerance: {optimal_tolerance}")
        print(f"При этом значении граф становится связным (все выработки соединены)")
    else:
        print(f"\nДаже при максимальном проверенном значении tolerance={max(tolerance_range)} граф не становится связным")
        print(f"Требуется дополнительный анализ или ручное добавление соединений")

if __name__ == "__main__":
    # Анализируем связность с разными параметрами tolerance
    identify_missing_connections([0.5, 1.0, 2.0, 5.0, 10.0]) 