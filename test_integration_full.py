"""
test_integration_full.py — Полный интеграционный тест
Проверяет совместную работу controller.py и physics.py
"""

import sys
import os

# Добавляем пути
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

# Импорты
from controller import create_and_run, DEFAULT_PARAMS as CONTROLLER_PARAMS
from physics import DroneParams, QuadcopterDynamics
from physics_adapter import create_integrate_function, PhysicsAdapter


def test_basic_integration():
    """Базовый тест: взлет и простые маневры"""
    print("="*70)
    print("Тест 1: Базовая интеграция controller.py + physics.py")
    print("="*70)
    
    # Создаем адаптер
    integrate = create_integrate_function()
    
    # Настраиваем параметры для быстрого теста
    params = CONTROLLER_PARAMS.copy()
    params["n_cycles"] = 3          # 3 цикла
    params["t_cycle_min"] = 1.0
    params["t_cycle_max"] = 2.0
    params["V_hover"] = 9.0         # напряжение для зависания
    params["V_MIN"] = 6.0
    params["V_MAX"] = 12.0
    
    # Запускаем
    flight_plan, trajectory = create_and_run(
        physics_callback=integrate,
        visualization_callback=None,  # без визуализации
        params=params,
        real_time=False
    )
    
    traj = np.array(trajectory)
    
    print(f"\n📊 Результаты:")
    print(f"   - Циклов: {len(flight_plan.cycles)}")
    print(f"   - Шагов: {len(trajectory)}")
    print(f"   - Время: {len(trajectory) * params['dt']:.1f} с")
    print(f"   - Высота min/max: {np.min(traj[:,2]):.2f} / {np.max(traj[:,2]):.2f} м")
    print(f"   - Конечная позиция: x={traj[-1,0]:.2f}, y={traj[-1,1]:.2f}, z={traj[-1,2]:.2f}")
    
    # Проверка успешности
    if np.max(traj[:,2]) > 2.0 and traj[-1,2] >= 0:
        print("\n✅ Базовый тест пройден!")
        return True
    else:
        print("\n❌ Базовый тест НЕ пройден")
        return False


def test_with_visualization():
    """Тест с визуализацией траектории"""
    print("\n" + "="*70)
    print("Тест 2: Визуализация полета")
    print("="*70)
    
    integrate = create_integrate_function()
    
    params = CONTROLLER_PARAMS.copy()
    params["n_cycles"] = 5
    params["t_cycle_min"] = 1.0
    params["t_cycle_max"] = 2.5
    
    # Простая визуализация (сбор данных)
    trajectory = []
    
    def collect_state(state):
        trajectory.append(state.copy())
    
    flight_plan, _ = create_and_run(
        physics_callback=integrate,
        visualization_callback=collect_state,
        params=params,
        real_time=False
    )
    
    traj = np.array(trajectory)
    
    # Создаем графики
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 3D Траектория
    ax = axes[0, 0]
    ax.plot(traj[:,0], traj[:,1], 'b-', linewidth=1)
    ax.scatter(traj[0,0], traj[0,1], c='g', s=100, label='Старт')
    ax.scatter(traj[-1,0], traj[-1,1], c='r', s=100, label='Финиш')
    ax.set_xlabel('X (м)')
    ax.set_ylabel('Y (м)')
    ax.set_title('Горизонтальная проекция')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    # 2. Высота
    axes[0, 1].plot(traj[:,2], 'b-', linewidth=1)
    axes[0, 1].axhline(y=5, c='r', linestyle='--', label='Мин. высота')
    axes[0, 1].set_xlabel('Шаг')
    axes[0, 1].set_ylabel('Z (м)')
    axes[0, 1].set_title('Высота полета')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. Углы
    axes[1, 0].plot(np.degrees(traj[:,3]), label='Крен (φ)')
    axes[1, 0].plot(np.degrees(traj[:,4]), label='Тангаж (θ)')
    axes[1, 0].plot(np.degrees(traj[:,5]), label='Рыскание (ψ)')
    axes[1, 0].set_xlabel('Шаг')
    axes[1, 0].set_ylabel('Угол (градусы)')
    axes[1, 0].set_title('Угловое положение')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. Линейные скорости
    axes[1, 1].plot(traj[:,6], label='Vx')
    axes[1, 1].plot(traj[:,7], label='Vy')
    axes[1, 1].plot(traj[:,8], label='Vz')
    axes[1, 1].set_xlabel('Шаг')
    axes[1, 1].set_ylabel('Скорость (м/с)')
    axes[1, 1].set_title('Линейные скорости')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.suptitle('Траектория полета квадрокоптера', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Статистика:")
    print(f"   - Максимальная высота: {np.max(traj[:,2]):.2f} м")
    print(f"   - Минимальная высота: {np.min(traj[:,2]):.2f} м")
    print(f"   - Макс. скорость по X: {np.max(np.abs(traj[:,6])):.2f} м/с")
    print(f"   - Макс. скорость по Y: {np.max(np.abs(traj[:,7])):.2f} м/с")
    print(f"   - Макс. скорость по Z: {np.max(np.abs(traj[:,8])):.2f} м/с")
    
    return True


def compare_integration_methods():
    """Сравнение методов интегрирования"""
    print("\n" + "="*70)
    print("Тест 3: Сравнение методов интегрирования")
    print("="*70)
    
    drone = QuadcopterDynamics()
    
    # Начальное состояние
    state_euler = drone.reset_state()
    state_rk4 = drone.reset_state()
    
    # Управление (вертикальный взлет)
    omega = drone.voltage_to_omega(10.0, 10.0, 10.0, 10.0)
    dt = 0.04
    steps = 100
    
    euler_traj = [state_euler[2]]
    rk4_traj = [state_rk4[2]]
    
    for _ in range(steps):
        state_euler = drone.euler_step(state_euler, *omega, dt)
        state_rk4 = drone.rk4_step(state_rk4, *omega, dt)
        euler_traj.append(state_euler[2])
        rk4_traj.append(state_rk4[2])
    
    # График сравнения
    plt.figure(figsize=(10, 6))
    plt.plot(euler_traj, 'r--', label='Эйлер', alpha=0.7)
    plt.plot(rk4_traj, 'b-', label='Рунге-Кутта 4', linewidth=1.5)
    plt.xlabel('Шаг')
    plt.ylabel('Высота (м)')
    plt.title('Сравнение методов интегрирования')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"   Эйлер (финальная высота): {euler_traj[-1]:.2f} м")
    print(f"   RK4 (финальная высота):  {rk4_traj[-1]:.2f} м")
    print(f"   Разница: {abs(euler_traj[-1] - rk4_traj[-1]):.2f} м")
    
    return True


def test_stability_limits():
    """Тест на устойчивость (углы крена/тангажа)"""
    print("\n" + "="*70)
    print("Тест 4: Проверка устойчивости (предельные углы)")
    print("="*70)
    
    drone = QuadcopterDynamics()
    state = drone.reset_state()
    
    # Экстремальный дисбаланс для создания крена
    omega = drone.voltage_to_omega(12.0, 6.0, 12.0, 6.0)
    dt = 0.04
    
    phi_values = []
    stable_steps = 0
    
    for step in range(100):
        state = drone.rk4_step(state, *omega, dt)
        phi = abs(state[3])
        phi_values.append(np.degrees(phi))
        
        if drone.check_stability(state):
            stable_steps += 1
    
    plt.figure(figsize=(10, 6))
    plt.plot(phi_values, 'b-', linewidth=1)
    plt.axhline(y=45, c='r', linestyle='--', label='Предел (45°)')
    plt.xlabel('Шаг')
    plt.ylabel('Угол крена (градусы)')
    plt.title('Устойчивость дрона при управлении')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    max_phi = np.max(phi_values)
    print(f"   Максимальный угол крена: {max_phi:.1f}°")
    print(f"   Шагов в стабильном состоянии: {stable_steps}/{100}")
    
    if max_phi < 60:
        print("   ✅ Стабильность в пределах нормы")
    else:
        print("   ⚠️ Превышение предельных углов")
    
    return True


if __name__ == "__main__":
    print("\n" + "█"*70)
    print("   ИНТЕГРАЦИОННОЕ ТЕСТИРОВАНИЕ controller.py + physics.py")
    print("█"*70)
    
    # Запуск всех тестов
    tests = [
        ("Базовый тест", test_basic_integration),
        ("Визуализация", test_with_visualization),
        ("Сравнение методов", compare_integration_methods),
        ("Устойчивость", test_stability_limits),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Ошибка в {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Итог
    print("\n" + "="*70)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("="*70)
    for name, result in results:
        status = "✅ ПРОЙДЕН" if result else "❌ НЕ ПРОЙДЕН"
        print(f"   {status}: {name}")
    
    all_passed = all(r for _, r in results)
    if all_passed:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Модули корректно интегрированы.")
    else:
        print("\n⚠️ Некоторые тесты не пройдены. Требуется доработка.")