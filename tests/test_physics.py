"""
test_physics.py - Тесты для физической модели
Запускать отдельно: python test_physics.py
"""

import sys
import os

# Добавляем корневую папку проекта в пути поиска (ДО импорта physics!)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from physics import DroneParams, QuadcopterDynamics


def test_vertical_takeoff():
    """Тест 1: Вертикальный взлет (стр. 7 пособия)"""
    print("\n=== Тест 1: Вертикальный взлет ===")
    
    drone = QuadcopterDynamics()
    state = drone.reset_state()
    
    omega1, omega2, omega3, omega4 = drone.voltage_to_omega(12.0, 12.0, 12.0, 12.0)
    dt = 0.04
    
    for step in range(50):
        state = drone.rk4_step(state, omega1, omega2, omega3, omega4, dt)
        if step % 10 == 0:
            x, y, z = drone.get_position(state)
            print(f"  t={step*dt:.2f}c: z={z:.2f}м, vz={state[8]:.2f}м/с")


def test_roll_control():
    print("\n=== Тест 2: Управление креном ===")
    
    drone = QuadcopterDynamics()
    state = drone.reset_state()
    dt = 0.04
    
    # Для крена нужна разница между ω2 и ω4
    # ω1 и ω3 — левые, ω2 и ω4 — правые
    omega_left = drone.voltage_to_omega(12.0, 9.0, 12.0, 10.0)   # левые 12V, правые 6V
    omega_right = drone.voltage_to_omega(6.0, 12.0, 6.0, 12.0)  # левые 6V, правые 12V
    omega_hover = drone.voltage_to_omega(9.0, 9.0, 9.0, 9.0)
    
    for step in range(150):
        if step < 50:
            omega1, omega2, omega3, omega4 = omega_left
        elif step < 100:
            omega1, omega2, omega3, omega4 = omega_hover
        else:
            omega1, omega2, omega3, omega4 = omega_right
        
        state = drone.rk4_step(state, omega1, omega2, omega3, omega4, dt)
        
        if step % 25 == 0:
            x, y, z = drone.get_position(state)
            phi, theta, psi = drone.get_attitude(state)
            print(f"  t={step*dt:.2f}c: x={x:.2f}м, z={z:.2f}м, φ={phi:.1f}°")
def test_control_delay():
    """Тест 3: Учет задержек управления (стр. 10-11)"""
    print("\n=== Тест 3: Учет задержек управления ===")
    
    drone = QuadcopterDynamics()
    state = drone.reset_state()
    current_omega = np.array([0.0, 0.0, 0.0, 0.0])
    target_omega = np.array(drone.voltage_to_omega(12.0, 12.0, 12.0, 12.0))
    delay_time = 0.2
    dt = 0.04
    
    for step in range(30):
        current_omega = drone.apply_control_with_delay(current_omega, target_omega, delay_time, dt)
        state = drone.rk4_step(state, current_omega[0], current_omega[1], current_omega[2], current_omega[3], dt)
        
        if step % 5 == 0:
            x, y, z = drone.get_position(state)
            print(f"  t={step*dt:.2f}c: ω1={current_omega[0]:.0f} рад/с, z={z:.2f}м")


if __name__ == "__main__":
    print("=" * 60)
    print("Тестирование физической модели квадрокоптера")
    print("=" * 60)
    
    test_vertical_takeoff()
    test_roll_control()
    test_control_delay()
    
    print("\n" + "=" * 60)
    print("Все тесты пройдены!")