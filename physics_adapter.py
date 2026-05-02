"""
physics_adapter.py — Адаптер для совместимости controller.py и physics.py

Преобразует вызов:
    new_state = physics_callback(state, omega_array, dt)
в вызов:
    new_state = drone.rk4_step(state, ω1, ω2, ω3, ω4, dt)
"""

import numpy as np
from physics import QuadcopterDynamics, DroneParams

class PhysicsAdapter:
    """
    Адаптер, предоставляющий интерфейс, ожидаемый controller.py.
    """
    
    def __init__(self, params=None):
        """
        Инициализация адаптера.
        
        Args:
            params: параметры дрона (DroneParams или словарь)
        """
        if params is None:
            self.drone = QuadcopterDynamics()
        elif isinstance(params, DroneParams):
            self.drone = QuadcopterDynamics(params)
        elif isinstance(params, dict):
            # Преобразование словаря в DroneParams
            drone_params = DroneParams(
                m_c=params.get('m_c', 0.692),
                m_r=params.get('m_r', 0.008),
                m_l=params.get('m_l', 0.094),
                r=params.get('r', 0.075),
                r_r=params.get('r_r', 0.12),
                l=params.get('l', 0.25),
                k=params.get('k', 1e-5),
                b=params.get('b', 1e-6),
                KV=params.get('KV', 1000.0),
                eta=params.get('eta', 0.7),
                V_MIN=params.get('V_MIN', 0.0),
                V_MAX=params.get('V_MAX', 12.0),
                g=params.get('g', 9.81)
            )
            self.drone = QuadcopterDynamics(drone_params)
        else:
            self.drone = QuadcopterDynamics()
        
        # Состояние дрона
        self.state = self.drone.reset_state()
    
    def integrate(self, state: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        """
        Основная функция для вызова из controller.py.
        
        Args:
            state: вектор состояния [x,y,z,φ,θ,ψ,vx,vy,vz,ωx,ωy,ωz]
            omega: массив угловых скоростей [ω1, ω2, ω3, ω4]
            dt: шаг интегрирования (с)
        
        Returns:
            new_state: обновленный вектор состояния
        """
        ω1, ω2, ω3, ω4 = omega[0], omega[1], omega[2], omega[3]
        
        # Используем RK4 метод для точности
        new_state = self.drone.rk4_step(state, ω1, ω2, ω3, ω4, dt)
        
        return new_state
    
    def reset(self) -> np.ndarray:
        """Сброс состояния в ноль."""
        self.state = self.drone.reset_state()
        return self.state
    
    def get_position(self, state=None):
        """Получить позицию дрона."""
        if state is None:
            state = self.state
        return self.drone.get_position(state)


# Функция для прямой передачи в controller.py
def create_integrate_function(params=None):
    """
    Создает функцию integrate, совместимую с controller.py.
    
    Использование:
        integrate = create_integrate_function()
        plan, trajectory = create_and_run(
            physics_callback=integrate,
            ...
        )
    """
    adapter = PhysicsAdapter(params)
    
    def integrate(state, omega, dt):
        return adapter.integrate(state, omega, dt)
    
    return integrate