import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_risk_data(file_path='risk_data.csv'):
    """Загрузка данных о рисках из CSV файла"""
    try:
        df = pd.read_csv(file_path)
        print(f"Загружены данные о рисках из {file_path}: {len(df)} записей")
        return df
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None

def create_risk_distribution(df, output_file='risk_distribution.png'):
    """Создание гистограммы распределения рисков"""
    print(f"Создание распределения рисков...")
    
    # Создаем подграфики
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Распределение рисков в шахтных выработках', fontsize=16)
    
    # График 1: Общее распределение рисков
    ax1 = axes[0, 0]
    ax1.hist(df['total_risk'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    
    # Добавление кривой плотности вероятности
    from scipy.stats import gaussian_kde
    data = df['total_risk']
    density = gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 1000)
    y_vals = density(x_vals)
    ax1.plot(x_vals, y_vals * len(data) * (data.max() - data.min()) / 20, color='darkblue', linewidth=2)
    
    ax1.set_title('Распределение полного риска')
    ax1.set_xlabel('Вероятность ЧП')
    ax1.set_ylabel('Количество выработок')
    
    # График 2: Сравнение компонентов риска
    ax2 = axes[0, 1]
    boxplot_data = [df['base_risk'], df['work_risk'], df['equipment_risk']]
    ax2.boxplot(boxplot_data, labels=['Базовый риск', 'Риск работ', 'Риск оборудования'])
    ax2.set_title('Сравнение компонентов риска')
    ax2.set_ylabel('Вероятность')
    
    # График 3: Топ-10 выработок с наибольшим риском
    ax3 = axes[1, 0]
    top_risks = df.sort_values('total_risk', ascending=False).head(10)
    colors = plt.cm.YlOrRd(top_risks['total_risk'] / top_risks['total_risk'].max())
    bars = ax3.bar(top_risks['node_id'], top_risks['total_risk'], color=colors)
    ax3.set_title('Топ-10 выработок с наибольшим риском')
    ax3.set_xlabel('Идентификатор выработки')
    ax3.set_ylabel('Полный риск')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # Добавляем текстовые метки со значениями
    for bar, risk in zip(bars, top_risks['total_risk']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{risk:.3f}', ha='center', va='bottom', fontsize=8)
    
    # График 4: Корреляция между риском и количеством работ/оборудования
    ax4 = axes[1, 1]
    top_risks['combined'] = top_risks['num_works'] + top_risks['num_equipment']
    colors = plt.cm.YlOrRd(top_risks['total_risk'] / top_risks['total_risk'].max())
    
    # Размер точек пропорционален риску
    sizes = 100 + 400 * (top_risks['total_risk'] / top_risks['total_risk'].max())
    
    scatter = ax4.scatter(top_risks['combined'], top_risks['total_risk'], 
                          s=sizes, c=colors, alpha=0.7, edgecolors='black')
    
    # Добавляем метки с названиями узлов
    for i, node_id in enumerate(top_risks['node_id']):
        ax4.annotate(node_id, 
                    (top_risks['combined'].iloc[i], top_risks['total_risk'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_title('Корреляция риска с количеством работ и оборудования')
    ax4.set_xlabel('Количество работ + оборудования')
    ax4.set_ylabel('Полный риск')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_file, dpi=300)
    print(f"Визуализация распределения рисков сохранена в {output_file}")
    
    return fig

def create_risk_map_by_status(df, output_file='risk_by_status.png'):
    """Создание графика рисков по статусу выработок"""
    print(f"Создание графика рисков по статусу...")
    
    # Группировка данных по статусу
    risk_by_status = df.groupby('status').agg({
        'total_risk': ['mean', 'min', 'max', 'count'],
        'base_risk': 'mean',
        'work_risk': 'mean',
        'equipment_risk': 'mean'
    }).reset_index()
    
    # Переименование колонок для удобства
    risk_by_status.columns = ['Статус', 'Ср. полный риск', 'Мин. риск', 'Макс. риск',
                           'Кол-во выработок', 'Ср. базовый риск', 'Ср. риск работ',
                           'Ср. риск оборудования']
    
    # Создание графика с двумя осями Y
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    color = 'tab:blue'
    ax1.set_xlabel('Статус выработки')
    ax1.set_ylabel('Средний риск', color=color)
    line1 = ax1.plot(risk_by_status['Статус'], risk_by_status['Ср. полный риск'], 
                    'o-', color=color, linewidth=2, markersize=10, label='Средний полный риск')
    line2 = ax1.plot(risk_by_status['Статус'], risk_by_status['Ср. базовый риск'], 
                    's--', color='tab:cyan', linewidth=2, markersize=8, label='Средний базовый риск')
    line3 = ax1.plot(risk_by_status['Статус'], risk_by_status['Ср. риск работ'], 
                    '^--', color='tab:green', linewidth=2, markersize=8, label='Средний риск работ')
    line4 = ax1.plot(risk_by_status['Статус'], risk_by_status['Ср. риск оборудования'], 
                    'd--', color='tab:purple', linewidth=2, markersize=8, label='Средний риск оборудования')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Добавление второй оси Y для количества выработок
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Количество выработок', color=color)
    bars = ax2.bar(risk_by_status['Статус'], risk_by_status['Кол-во выработок'], 
                  alpha=0.3, color=color, label='Количество выработок')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Добавление текстовых аннотаций над столбцами
    for i, count in enumerate(risk_by_status['Кол-во выработок']):
        ax2.annotate(f"{count}", 
                    xy=(risk_by_status['Статус'][i], count),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, color='black')
    
    # Добавление легенды
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Добавление заголовка и сетки
    plt.title('Риски и количество выработок по статусу', fontsize=16)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Визуализация рисков по статусу сохранена в {output_file}")
    
    return fig

def create_total_risk_pie(df, output_file='total_risk_pie.png'):
    """Создание круговой диаграммы для оценки интегрального риска"""
    print(f"Создание круговой диаграммы интегрального риска...")
    
    # Рассчитываем интегральный риск (вероятность хотя бы одного ЧП)
    no_incident_prob = 1.0
    for risk in df['total_risk']:
        no_incident_prob *= (1.0 - risk)
    
    total_system_risk = 1.0 - no_incident_prob
    
    # Создаем данные для круговой диаграммы
    labels = ['Вероятность ЧП', 'Вероятность отсутствия ЧП']
    sizes = [total_system_risk, 1 - total_system_risk]
    explode = (0.1, 0)  # выделяем первый сегмент
    colors = ['#FF6B6B', '#4ECDC4']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%',
                                    shadow=True, startangle=90, textprops={'fontsize': 14})
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    plt.title(f'Интегральная вероятность ЧП в шахте: {total_system_risk:.4f}', fontsize=16)
    
    # Добавляем аннотацию с дополнительной информацией
    plt.annotate(
        f"Количество выработок: {len(df)}\n"
        f"Средний риск по выработкам: {df['total_risk'].mean():.4f}\n"
        f"Максимальный риск: {df['total_risk'].max():.4f}",
        xy=(0.5, 0.03), xycoords='figure fraction',
        ha='center', bbox=dict(boxstyle="round,pad=0.5", fc='#F0F0F0', alpha=0.8),
        fontsize=12
    )
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Круговая диаграмма интегрального риска сохранена в {output_file}")
    
    return fig

def main():
    print("="*80)
    print("ВИЗУАЛИЗАЦИЯ СТАТИСТИКИ РИСКОВ В ШАХТНОЙ СИСТЕМЕ")
    print("="*80)
    
    # Проверяем и создаем директорию для сохранения результатов
    output_dir = "risk_stats"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана директория {output_dir} для сохранения результатов")
    
    # Устанавливаем matplotlib в неинтерактивный режим
    plt.ioff()
    
    # Загрузка данных о рисках
    df = load_risk_data()
    if df is None:
        print("Невозможно продолжить без данных о рисках. Убедитесь, что файл risk_data.csv существует.")
        print("Сначала выполните скрипт risk_analyzer.py для создания этого файла.")
        return False
    
    print("\nСоздание визуализаций...")
    # Создание различных визуализаций
    try:
        create_risk_distribution(df, os.path.join(output_dir, 'risk_distribution.png'))
        print("✓ Распределение рисков создано")
    except Exception as e:
        print(f"✗ Ошибка при создании распределения рисков: {e}")
    
    try:
        create_risk_map_by_status(df, os.path.join(output_dir, 'risk_by_status.png'))
        print("✓ График рисков по статусу создан")
    except Exception as e:
        print(f"✗ Ошибка при создании графика рисков по статусу: {e}")
    
    try:
        create_total_risk_pie(df, os.path.join(output_dir, 'total_risk_pie.png'))
        print("✓ Круговая диаграмма интегрального риска создана")
    except Exception as e:
        print(f"✗ Ошибка при создании круговой диаграммы: {e}")
    
    # Также сохраняем копии в корневой директории для совместимости
    try:
        create_risk_distribution(df, 'risk_distribution.png')
        create_risk_map_by_status(df, 'risk_by_status.png')
        create_total_risk_pie(df, 'total_risk_pie.png')
    except Exception as e:
        print(f"Предупреждение: не удалось создать копии файлов в корневой директории: {e}")
    
    print("\nВсе визуализации успешно созданы!")
    print("Результаты сохранены в директории:", output_dir)
    print("  - risk_distribution.png - Распределение рисков в выработках")
    print("  - risk_by_status.png - Риски по статусу выработок")
    print("  - total_risk_pie.png - Интегральная вероятность ЧП")
    print("="*80)
    
    # Закрываем все открытые фигуры matplotlib
    plt.close('all')
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\nВизуализация статистики рисков успешно завершена.")
    else:
        print("\nВизуализация завершена с ошибками.") 