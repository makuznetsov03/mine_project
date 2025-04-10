import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import csv

# Отключаем интерактивное отображение графиков
plt.ioff()  # отключает интерактивный режим - графики не будут показываться

def load_data():
    """Загрузка данных из CSV файлов"""
    print("Загрузка данных...")
    mine_axes = pd.read_csv('mine_axes.csv')
    equipment = pd.read_csv('equipment.csv')
    axis_works = pd.read_csv('axis_works.csv')
    works = pd.read_csv('works.csv')
    
    return mine_axes, equipment, axis_works, works

def build_risk_graph(mine_axes, equipment, axis_works, works, tolerance=2.0):
    """Построение графа с информацией о рисках и вероятностях ЧП"""
    print("Построение графа с информацией о рисках...")
    G = nx.Graph()
    
    # Добавление узлов (выработок) с базовыми вероятностями ЧП
    for idx, row in mine_axes.iterrows():
        node_id = row['short_name']
        
        # Базовая вероятность ЧП для выработки (зависит от статуса выработки)
        # Уменьшаем базовый риск для снижения интегральной вероятности
        base_risk = 0.0005 * row['status']  # Уменьшили в 2 раза с 0.001
        
        G.add_node(node_id, 
                  name=row['full_name'],
                  status=row['status'],
                  act_works=row['act_works'],
                  start_pos=(row['xs'], row['ys'], row['zs']),
                  end_pos=(row['xf'], row['yf'], row['zf']),
                  length=np.sqrt((row['xf']-row['xs'])**2 + (row['yf']-row['ys'])**2 + (row['zf']-row['zs'])**2),
                  equipment=[],
                  works=[],
                  base_risk=base_risk,
                  total_risk=base_risk)  # Инициализируем полную вероятность базовой
    
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
                    G.add_edge(node1, node2, weight=distance, edge_risk=distance/100.0)  # Пример: риск пропорционален расстоянию
    
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
                    pos1 = G.nodes[node1]['start_pos']
                    
                    for node2 in comp2:
                        pos2 = G.nodes[node2]['start_pos']
                        
                        distance = np.sqrt(sum((a-b)**2 for a, b in zip(pos1, pos2)))
                        if distance < min_distance:
                            min_distance = distance
                            closest_nodes = (node1, node2)
                
                # Добавляем ребро между ближайшими узлами
                if closest_nodes:
                    node1, node2 = closest_nodes
                    G.add_edge(node1, node2, weight=min_distance, artificial=True, edge_risk=min_distance/50.0)  # Повышенный риск для искусственных соединений
                    print(f"Добавлено искусственное соединение между {node1} и {node2} с расстоянием {min_distance:.2f}")
    
    # Связывание работ с выработками и расчет рисков
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
            
            work_risk_sum = 0.0  # Суммарный риск от работ
            
            for code in work_codes:
                work_data = works[works['work_code'] == code]
                if not work_data.empty:
                    # Извлекаем риск работы из данных
                    work_risk = work_data.iloc[0]['ud_risk']
                    work_risk_sum += float(work_risk)
                    
                    workers_info.append({
                        'work_code': code,
                        'name': work_data.iloc[0]['full_name'],
                        'risk': work_risk,
                        'color': work_data.iloc[0]['col_work']
                    })
            
            G.nodes[node]['works'] = workers_info
            
            # Обновляем вероятность ЧП с учетом работ
            base_risk = G.nodes[node]['base_risk']
            G.nodes[node]['work_risk'] = work_risk_sum
            G.nodes[node]['total_risk'] = calculate_node_risk(base_risk, work_risk_sum, 0.0)  # Пока без оборудования
    
    # Привязка оборудования к выработкам и расчет рисков
    for _, row in equipment.iterrows():
        eq_id = row['short_name']
        eq_start = (row['xs'], row['ys'], row['zs'])
        eq_end = (row['xf'], row['yf'], row['zf'])
        
        # Находим ближайшую выработку к оборудованию
        min_distance = float('inf')
        closest_node = None
        
        for node in G.nodes:
            node_start = G.nodes[node]['start_pos']
            node_end = G.nodes[node]['end_pos']
            
            # Вычисляем расстояние между центральными точками
            eq_center = ((eq_start[0] + eq_end[0]) / 2, 
                         (eq_start[1] + eq_end[1]) / 2, 
                         (eq_start[2] + eq_end[2]) / 2)
            
            node_center = ((node_start[0] + node_end[0]) / 2, 
                           (node_start[1] + node_end[1]) / 2, 
                           (node_start[2] + node_end[2]) / 2)
            
            distance = np.sqrt(sum((a-b)**2 for a, b in zip(eq_center, node_center)))
            
            if distance < min_distance:
                min_distance = distance
                closest_node = node
        
        # Добавляем оборудование к ближайшей выработке
        if closest_node and min_distance < tolerance * 2:
            # Риск от оборудования зависит от его статуса и типа
            # Снижаем риск от оборудования
            eq_risk = 0.001 * row['status'] * (1 + row['line_eq'] / 10)  # Уменьшено с 0.002 до 0.001
            
            eq_info = {
                'id': eq_id,
                'name': row['full_name'],
                'status': row['status'],
                'type': row['line_eq'],
                'start_pos': eq_start,
                'end_pos': eq_end,
                'risk': eq_risk
            }
            
            G.nodes[closest_node]['equipment'].append(eq_info)
            
            # Обновляем общую вероятность ЧП
            base_risk = G.nodes[closest_node]['base_risk']
            work_risk = G.nodes[closest_node].get('work_risk', 0.0)
            
            # Собираем все риски оборудования
            eq_risk_sum = sum(e['risk'] for e in G.nodes[closest_node]['equipment'])
            G.nodes[closest_node]['equipment_risk'] = eq_risk_sum
            
            G.nodes[closest_node]['total_risk'] = calculate_node_risk(base_risk, work_risk, eq_risk_sum)
    
    return G

def calculate_node_risk(base_risk, work_risk, equipment_risk):
    """
    Формула для расчета полной вероятности ЧП в выработке.
    
    Модифицированная формула для снижения итоговой вероятности:
    total_risk = base_risk + work_risk*0.7 + equipment_risk*0.6 - 
                 (base_risk * work_risk*0.7) - (base_risk * equipment_risk*0.6) - 
                 (work_risk*0.7 * equipment_risk*0.6) + 
                 (base_risk * work_risk*0.7 * equipment_risk*0.6)
    
    Введены понижающие коэффициенты для рисков от работ и оборудования.
    """
    # Применяем понижающие коэффициенты
    work_risk = work_risk * 0.7
    equipment_risk = equipment_risk * 0.6
    
    # Вероятность того, что хотя бы одно из событий произойдет (Правило сложения вероятностей)
    total_risk = base_risk + work_risk + equipment_risk
    
    # Вычитаем вероятности пересечений (чтобы не учитывать дважды)
    total_risk -= base_risk * work_risk
    total_risk -= base_risk * equipment_risk
    total_risk -= work_risk * equipment_risk
    
    # Добавляем вероятность пересечения всех трех событий
    total_risk += base_risk * work_risk * equipment_risk
    
    # Ограничиваем вероятность значениями между 0 и 1
    return min(max(total_risk, 0.0), 1.0)

def analyze_risks(G):
    """Анализ рисков в графе шахтных выработок"""
    print("\nАнализ рисков шахтных выработок:")
    
    # Общая статистика по рискам
    risks = [data['total_risk'] for _, data in G.nodes(data=True)]
    avg_risk = np.mean(risks)
    max_risk = np.max(risks)
    min_risk = np.min(risks)
    
    print(f"Средняя вероятность ЧП: {avg_risk:.6f}")
    print(f"Максимальная вероятность ЧП: {max_risk:.6f}")
    print(f"Минимальная вероятность ЧП: {min_risk:.6f}")
    
    # Выработки с наибольшим риском
    high_risk_nodes = sorted(G.nodes(data=True), key=lambda x: x[1]['total_risk'], reverse=True)
    
    print("\nТоп-5 выработок с наивысшей вероятностью ЧП:")
    for i, (node, data) in enumerate(high_risk_nodes[:5]):
        print(f"{i+1}. {node} ({data['name']}): {data['total_risk']:.6f}")
        print(f"   - Базовый риск: {data['base_risk']:.6f}")
        print(f"   - Риск от работ: {data.get('work_risk', 0.0):.6f}")
        print(f"   - Риск от оборудования: {data.get('equipment_risk', 0.0):.6f}")
        
        # Выводим информацию о работах
        if 'works' in data and data['works']:
            print(f"   - Работы ({len(data['works'])}):")
            for work in data['works']:
                print(f"     * {work['name']} (код {work['work_code']}): риск {work['risk']}")
        else:
            print("   - Нет работ")
            
        # Выводим информацию об оборудовании
        if 'equipment' in data and data['equipment']:
            print(f"   - Оборудование ({len(data['equipment'])}):")
            for eq in data['equipment']:
                print(f"     * {eq['name']} (id {eq['id']}): риск {eq['risk']:.6f}")
        else:
            print("   - Нет оборудования")
    
    # Расчет полной вероятности ЧП по всей шахте
    # (вероятность того, что хотя бы в одной выработке произойдет ЧП)
    total_system_risk = calculate_system_risk(risks)
    print(f"\nИнтегральная вероятность ЧП по всей шахте: {total_system_risk:.6f}")
    
    return high_risk_nodes

def calculate_system_risk(node_risks):
    """
    Расчет интегральной вероятности ЧП в шахте.
    Вероятность того, что хотя бы в одной выработке произойдет ЧП.
    
    P(хотя бы одно ЧП) = 1 - P(ни одного ЧП)
    P(ни одного ЧП) = Произведение (1 - P(ЧП в выработке i))
    """
    no_incident_prob = 1.0
    for risk in node_risks:
        no_incident_prob *= (1.0 - risk)
    
    return 1.0 - no_incident_prob

def visualize_risk_graph(G, save_path='risk_map.png'):
    """Визуализация графа шахтных выработок с учетом рисков"""
    print(f"\nСоздание визуализации графа рисков...")
    
    # Создаем фигуру с двумя подграфиками - 3D и 2D
    fig = plt.figure(figsize=(20, 12))
    
    # 3D-визуализация
    ax1 = fig.add_subplot(121, projection='3d')
    
    # 2D-визуализация (вид сбоку: x-z плоскость)
    ax2 = fig.add_subplot(122)
    
    # Позиции узлов
    pos_3d = {}
    for node in G.nodes:
        start_pos = G.nodes[node]['start_pos']
        end_pos = G.nodes[node]['end_pos']
        pos_3d[node] = (
            (start_pos[0] + end_pos[0]) / 2,
            (start_pos[1] + end_pos[1]) / 2,
            (start_pos[2] + end_pos[2]) / 2
        )
    
    # Отрисовка выработок как линий в 3D
    for node in G.nodes:
        start_x, start_y, start_z = G.nodes[node]['start_pos']
        end_x, end_y, end_z = G.nodes[node]['end_pos']
        ax1.plot([start_x, end_x], [start_y, end_y], [start_z, end_z], 'k-', linewidth=1, alpha=0.3)
    
    # Определение цветов узлов на основе вероятности ЧП
    node_colors = []
    node_sizes = []
    for node in G.nodes:
        risk = G.nodes[node]['total_risk']
        
        # Цвет от зеленого (низкий риск) до красного (высокий риск)
        if risk < 0.01:  # Очень низкий риск
            node_colors.append('green')
        elif risk < 0.05:  # Низкий риск
            node_colors.append('yellowgreen')
        elif risk < 0.1:  # Средний риск
            node_colors.append('yellow')
        elif risk < 0.2:  # Высокий риск
            node_colors.append('orange')
        else:  # Очень высокий риск
            node_colors.append('red')
        
        # Размер узла зависит от риска
        node_sizes.append(50 + 500 * risk)
    
    # Координаты узлов для 3D
    x_nodes = [pos_3d[node][0] for node in G.nodes]
    y_nodes = [pos_3d[node][1] for node in G.nodes]
    z_nodes = [pos_3d[node][2] for node in G.nodes]
    
    # Отрисовка узлов в 3D
    scatter1 = ax1.scatter(x_nodes, y_nodes, z_nodes, c=node_colors, s=node_sizes, alpha=0.7, edgecolors='black')
    
    # Подписи узлов с информацией о риске в 3D
    for i, node in enumerate(G.nodes):
        x, y, z = pos_3d[node]
        risk_text = f"{node}: {G.nodes[node]['total_risk']:.4f}"
        ax1.text(x, y, z, risk_text, fontsize=8)
    
    # Отрисовка соединений между выработками в 3D
    for edge in G.edges:
        node1, node2 = edge
        x1, y1, z1 = pos_3d[node1]
        x2, y2, z2 = pos_3d[node2]
        
        # Стиль линии зависит от типа соединения
        if G.edges[edge].get('artificial', False):
            ax1.plot([x1, x2], [y1, y2], [z1, z2], 'r--', linewidth=1.5, alpha=0.7)
        else:
            ax1.plot([x1, x2], [y1, y2], [z1, z2], 'b--', linewidth=0.5, alpha=0.2)
    
    # Настройка осей и заголовка 3D
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Карта рисков шахтных выработок', fontsize=14)
    
    # 2D визуализация (вид сбоку: x-z плоскость)
    
    # Отрисовка узлов (центров выработок) в 2D
    for node in G.nodes:
        risk = G.nodes[node]['total_risk']
        
        # Цвет от зеленого (низкий риск) до красного (высокий риск)
        if risk < 0.01:  # Очень низкий риск
            color = 'green'
        elif risk < 0.05:  # Низкий риск
            color = 'yellowgreen'
        elif risk < 0.1:  # Средний риск
            color = 'yellow'
        elif risk < 0.2:  # Высокий риск
            color = 'orange'
        else:  # Очень высокий риск
            color = 'red'
        
        # Размер узла зависит от риска
        size = 50 + 500 * risk
        
        # Рисуем узел в центре выработки
        x = pos_3d[node][0]
        z = pos_3d[node][2]
        ax2.scatter(x, z, color=color, s=size, alpha=0.7, edgecolors='black')
        
        # Подпись узла
        risk_text = f"{node}: {risk:.4f}"
        ax2.text(x, z, risk_text, fontsize=8)
    
    # Отрисовка соединений между выработками в 2D
    for edge in G.edges:
        node1, node2 = edge
        x1, z1 = pos_3d[node1][0], pos_3d[node1][2]
        x2, z2 = pos_3d[node2][0], pos_3d[node2][2]
        
        # Стиль линии зависит от типа соединения
        if G.edges[edge].get('artificial', False):
            ax2.plot([x1, x2], [z1, z2], 'r--', linewidth=1.5, alpha=0.7)
        else:
            ax2.plot([x1, x2], [z1, z2], 'b--', linewidth=0.8, alpha=0.5)
    
    # Настройка осей и заголовка 2D
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title('Граф рисков шахты - схематический вид сбоку (X-Z)', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Легенда для рисков
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Очень низкий риск (<0.01)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='yellowgreen', markersize=10, label='Низкий риск (0.01-0.05)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Средний риск (0.05-0.1)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Высокий риск (0.1-0.2)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Очень высокий риск (>0.2)')
    ]
    
    # Добавляем общую легенду для обоих графиков
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.02))
    
    # Сохранение и отображение
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Оставляем место для легенды
    plt.savefig(save_path, dpi=300)
    print(f"Визуализация сохранена в файл: {save_path}")
    
    return fig, (ax1, ax2)

def save_risk_data(G, output_file='risk_data.csv'):
    """Сохранение данных о рисках в CSV файл"""
    print(f"\nСохранение данных о рисках в файл {output_file}...")
    
    # Подготовка данных
    risk_data = []
    for node, data in G.nodes(data=True):
        node_data = {
            'node_id': node,
            'name': data['name'],
            'status': data['status'],
            'base_risk': data['base_risk'],
            'work_risk': data.get('work_risk', 0.0),
            'equipment_risk': data.get('equipment_risk', 0.0),
            'total_risk': data['total_risk'],
            'num_works': len(data.get('works', [])),
            'num_equipment': len(data.get('equipment', [])),
        }
        risk_data.append(node_data)
    
    # Сохранение в CSV
    risk_df = pd.DataFrame(risk_data)
    risk_df.to_csv(output_file, index=False)
    print(f"Данные о рисках сохранены в файл: {output_file}")
    
    return risk_df

def main():
    # Загрузка данных
    mine_axes, equipment, axis_works, works = load_data()
    
    # Построение графа с информацией о рисках
    G = build_risk_graph(mine_axes, equipment, axis_works, works)
    
    # Анализ рисков
    high_risk_nodes = analyze_risks(G)
    
    # Визуализация графа с учетом рисков
    visualize_risk_graph(G)
    
    # Сохранение данных о рисках в CSV
    save_risk_data(G)
    
    print("\nАнализ рисков шахтных выработок завершен!")
    return G

if __name__ == "__main__":
    G = main()
    
    # Выводим интегральную вероятность ЧП для отладки
    risks = [data['total_risk'] for _, data in G.nodes(data=True)]
    total_system_risk = calculate_system_risk(risks)
    print(f"\n=============================================================")
    print(f"ИТОГОВАЯ ИНТЕГРАЛЬНАЯ ВЕРОЯТНОСТЬ ЧП: {total_system_risk:.6f}")
    print(f"=============================================================\n") 