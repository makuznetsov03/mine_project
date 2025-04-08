import os
import subprocess
import time
import sys

def check_dependencies():
    """Проверка и установка зависимостей"""
    try:
        import pandas as pd
        print("✓ pandas установлен")
    except ImportError:
        print("⚠ pandas не установлен")
        install = input("Установить pandas? (y/n): ")
        if install.lower() == 'y':
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        else:
            print("⚠ pandas требуется для работы программы")
    
    try:
        import networkx as nx
        print("✓ networkx установлен")
    except ImportError:
        print("⚠ networkx не установлен")
        install = input("Установить networkx? (y/n): ")
        if install.lower() == 'y':
            subprocess.check_call([sys.executable, "-m", "pip", "install", "networkx"])
        else:
            print("⚠ networkx требуется для работы программы")
    
    try:
        import matplotlib
        print("✓ matplotlib установлен")
    except ImportError:
        print("⚠ matplotlib не установлен")
        install = input("Установить matplotlib? (y/n): ")
        if install.lower() == 'y':
            subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
        else:
            print("⚠ matplotlib требуется для работы программы")
    
    try:
        import numpy
        print("✓ numpy установлен")
    except ImportError:
        print("⚠ numpy не установлен")
        install = input("Установить numpy? (y/n): ")
        if install.lower() == 'y':
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
        else:
            print("⚠ numpy требуется для работы программы")

def run_script(script_name, description):
    """Запуск скрипта с сообщением"""
    print(f"\n{'-'*40}")
    print(f"Выполнение: {description}")
    print(f"{'-'*40}")
    
    try:
        start_time = time.time()
        subprocess.check_call([sys.executable, script_name])
        end_time = time.time()
        print(f"Выполнено за {end_time - start_time:.2f} секунд")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении скрипта {script_name}:")
        print(f"  {e}")
        return False
    except Exception as e:
        print(f"Неизвестная ошибка при выполнении скрипта {script_name}:")
        print(f"  {e}")
        return False

def main():
    print("="*80)
    print("МОДЕЛИРОВАНИЕ ШАХТНОЙ СИСТЕМЫ")
    print("="*80)
    print("Данный проект создает модель шахтной системы с использованием графов.")
    print("Программа выполнит следующие шаги:")
    print("1. Построение и визуализация графа шахтных выработок")
    print("2. Моделирование перемещения рабочих по шахте")
    print("3. Анализ рисков шахтной системы")
    print("="*80)
    
    # Проверка наличия необходимых файлов
    required_files = ['mine_axes.csv', 'equipment.csv', 'axis_works.csv', 'works.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("ОШИБКА: Отсутствуют следующие файлы данных:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nПожалуйста, убедитесь, что все необходимые файлы находятся в текущей директории.")
        return False
    
    # Проверка и установка зависимостей
    print("\nПроверка зависимостей:")
    check_dependencies()
    
    # Выполнение скриптов
    if not run_script('mine_network.py', "Построение и визуализация графа шахтных выработок"):
        return False
    
    print("\nХотите запустить анимацию перемещения работников? (y/n)")
    choice = input("> ").strip().lower()
    
    if choice == 'y':
        # Проверка наличия ffmpeg для сохранения анимации
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("✓ ffmpeg установлен, можно создавать анимацию")
        except FileNotFoundError:
            print("⚠ ffmpeg не найден. Анимация будет только отображена, но не сохранена в файл.")
            print("Для сохранения анимации установите ffmpeg: https://ffmpeg.org/download.html")
        
        print("\nКакой тип анимации вы хотите запустить?")
        print("1. Стандартная анимация (mine_simulation.py)")
        print("2. Быстрая анимация (quick_simulation.py)")
        print("3. Создание видео из кадров (create_animation.py)")
        ani_choice = input("> ").strip()
        
        if ani_choice == '1':
            run_script('mine_simulation.py', "Анимация перемещения работников по шахте")
        elif ani_choice == '2':
            run_script('quick_simulation.py', "Быстрая анимация перемещения работников по шахте")
        elif ani_choice == '3':
            run_script('create_animation.py', "Создание видео перемещения работников")
        else:
            print("Некорректный выбор, запускаем стандартную анимацию...")
            run_script('mine_simulation.py', "Анимация перемещения работников по шахте")
    
    print("\nХотите запустить анализ рисков шахтной системы? (y/n)")
    risk_choice = input("> ").strip().lower()
    
    if risk_choice == 'y':
        run_script('risk_analyzer.py', "Анализ рисков шахтной системы")
        
        print("\nХотите создать статистические графики по рискам? (y/n)")
        stats_choice = input("> ").strip().lower()
        
        if stats_choice == 'y':
            run_script('risk_stats.py', "Статистический анализ рисков")
    
    print("\n"+"="*80)
    print("РЕЗУЛЬТАТЫ РАБОТЫ:")
    print("  - mine_network_axes.csv - Данные графа шахтных выработок")
    print("  - mine_network_edges.csv - Данные о смежностях выработок")
    print("  - mine_network.png - Визуализация графа шахтных выработок")
    if choice == 'y':
        if 'ani_choice' in locals():
            if ani_choice == '1':
                print("  - workers_animation.mp4 - Анимация перемещения работников (если установлен ffmpeg)")
            elif ani_choice == '2':
                print("  - workers_animation.gif - GIF-анимация перемещения работников")
                print("  - quick_frames/ - Кадры для быстрой анимации")
            elif ani_choice == '3':
                print("  - workers_animation.mp4 - Видео перемещения работников")
                print("  - animation_frames/ - Кадры для анимации")
        else:
            print("  - workers_animation.mp4 - Анимация перемещения работников (если установлен ffmpeg)")
        print("  - workers_data.csv - Данные о перемещении работников")
    if risk_choice == 'y':
        print("  - risk_map.png - Карта рисков шахтных выработок")
        print("  - risk_data.csv - Данные о рисках в выработках")
        if stats_choice == 'y':
            print("  - risk_stats/ - Директория с визуализациями статистики рисков:")
            print("    - risk_distribution.png - Распределение рисков в выработках")
            print("    - risk_by_status.png - Риски по статусу выработок")
            print("    - total_risk_pie.png - Интегральная вероятность ЧП")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\nПрограмма успешно завершена.")
    else:
        print("\nПрограмма завершена с ошибками.") 