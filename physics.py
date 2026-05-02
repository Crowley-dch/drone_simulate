"""
physics.py - Модуль физической модели движения квадрокоптера

Полная реализация в соответствии с пособием:
- Тельный А.В., Монахов М.Ю. Имитационная модель движения беспилотного 
  летательного аппарата типа квадрокоптер. 2025.

Реализованные формулы:
(1) Система дифференциальных уравнений движения
(2) Коэффициенты a1...a8
(3) Перевод напряжения в угловую скорость винтов
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional


@dataclass
class DroneParams:
    """
    Параметры дрона.
    Таблица 1 (стр. 9 пособия) + дополнительные параметры.
    """
    # === Массогабаритные характеристики (Таблица 1) ===
    m_c: float = 0.692      # масса корпуса БпЛА, кг
    m_r: float = 0.008      # масса одного винта БпЛА, кг
    m_l: float = 0.094      # масса луча БпЛА с двигателем, винтом и регулятором, кг
    r: float = 0.075        # радиус корпуса, м
    r_r: float = 0.12       # радиус винта, м
    l: float = 0.225        # длина луча БпЛА, м
    
    # === Аэродинамические коэффициенты (ФИНАЛЬНЫЕ) ===
    k: float = 1e-6         # коэффициент аэродинамического сопротивления
    b: float = 3e-9         # коэффициент тяги БпЛА (подобран для реалистичной скорости)
    
    # === Параметры двигателей ===
    KV: float = 920.0       # количество оборотов на вольт в минуту
    eta: float = 0.7        # КПД под нагрузкой (η = 0,6…0,8)
    V_MIN: float = 0.0      # минимальное напряжение на моторе, В
    V_MAX: float = 12.0     # максимальное напряжение на моторе, В
    
    # === Прочие физические константы ===
    g: float = 9.81         # ускорение свободного падения, м/с²
    
    # === Защита от сингулярности ===
    epsilon_angle: float = 1e-6   # для защиты от деления на ноль при cosθ → 0
    
    # === Вычисляемые параметры (заполняются в __post_init__) ===
    m: float = field(default=0.0, init=False)        # общая масса БпЛА, кг
    Jx: float = field(default=0.0, init=False)       # момент инерции по оси X, кг·м²
    Jy: float = field(default=0.0, init=False)       # момент инерции по оси Y, кг·м²
    Jz: float = field(default=0.0, init=False)       # момент инерции по оси Z, кг·м²
    Jr: float = field(default=0.0, init=False)       # момент инерции винта, кг·м²
    
    def __post_init__(self):
        """
        Расчет общей массы и моментов инерции (стр. 8-9 пособия).
        """
        # Общая масса: корпус + 4 луча + 4 винта
        self.m = self.m_c + 4 * self.m_l + 4 * self.m_r
        
        # Масса одного луча с винтом
        m_arm = self.m_l + self.m_r
        
        # Моменты инерции рамы (точечные массы на концах лучей)
        self.Jx = 2.0 * m_arm * self.l ** 2
        self.Jy = 2.0 * m_arm * self.l ** 2
        self.Jz = 4.0 * m_arm * self.l ** 2
        
        # Момент инерции винта (диск)
        self.Jr = 0.5 * self.m_r * self.r_r ** 2


class QuadcopterDynamics:
    """
    Класс, реализующий динамику квадрокоптера.
    
    Полная реализация формул (1), (2), (3) из пособия.
    """
    
    def __init__(self, params: Optional[DroneParams] = None):
        """
        Инициализация модели динамики.
        
        Args:
            params: параметры дрона. Если не указаны, используются стандартные.
        """
        self.p = params if params is not None else DroneParams()
        
        # Предрасчет коэффициентов a1...a8 по формуле (2)
        self._compute_coefficients()
        
        # Параметры стабильности
        self.max_roll_pitch_deg = 50.0   # максимальные углы крена/тангажа, градусы
        self.max_roll_pitch_rad = np.radians(self.max_roll_pitch_deg)
        
        # Коэффициенты демпфирования
        self.damping_roll = 3.5
        self.damping_pitch = 3.5
        self.damping_yaw = 2
        self.stability_factor = 0.6
        self.linear_drag = 0.1
    
    def _compute_coefficients(self) -> None:
        """
        Расчет коэффициентов a1...a8 по формуле (2) пособия (стр. 8).
        """
        self.a1 = self.p.k / self.p.m
        self.a2 = self.p.g
        self.a3 = (self.p.Jy - self.p.Jz) / self.p.Jx
        self.a4 = self.p.Jr / self.p.Jx
        self.a5 = (self.p.l * self.p.k) / self.p.Jx
        self.a6 = (self.p.Jz - self.p.Jx) / self.p.Jy
        self.a7 = (self.p.Jx - self.p.Jy) / self.p.Jz
        self.a8 = self.p.b / self.p.Jz
    
    def voltage_to_omega(self, U1: float, U2: float, U3: float, U4: float) -> Tuple[float, float, float, float]:
        """
        Формула (3) пособия (стр. 9):
        ω₁₋₄ = (U_p · KV · η · 2π) / 60
        
        Args:
            U1, U2, U3, U4: напряжение на каждом двигателе, В
        
        Returns:
            ω1, ω2, ω3, ω4: угловые скорости винтов, рад/с
        """
        # Ограничение напряжения согласно стр. 9
        U1 = np.clip(U1, self.p.V_MIN, self.p.V_MAX)
        U2 = np.clip(U2, self.p.V_MIN, self.p.V_MAX)
        U3 = np.clip(U3, self.p.V_MIN, self.p.V_MAX)
        U4 = np.clip(U4, self.p.V_MIN, self.p.V_MAX)
        
        # Коэффициент перевода: (KV * η * 2π) / 60
        factor = (self.p.KV * self.p.eta * 6.283185307179586) / 60.0
        
        omega1 = U1 * factor
        omega2 = U2 * factor
        omega3 = U3 * factor
        omega4 = U4 * factor
        
        return omega1, omega2, omega3, omega4
    
    def compute_accelerations(self, state: np.ndarray, 
                              omega1: float, omega2: float, 
                              omega3: float, omega4: float) -> np.ndarray:
        """
        Вычисление производных состояния по формуле (1) пособия (стр. 7-8).
        
        Вектор состояния (12 компонент):
        state[0]  = x   - координата по оси X, м
        state[1]  = y   - координата по оси Y, м
        state[2]  = z   - координата по оси Z (высота), м
        state[3]  = φ   - угол крена (roll), рад
        state[4]  = θ   - угол тангажа (pitch), рад
        state[5]  = ψ   - угол рыскания (yaw), рад
        state[6]  = vx  - линейная скорость по X, м/с
        state[7]  = vy  - линейная скорость по Y, м/с
        state[8]  = vz  - линейная скорость по Z, м/с
        state[9]  = ωx  - угловая скорость крена, рад/с
        state[10] = ωy  - угловая скорость тангажа, рад/с
        state[11] = ωz  - угловая скорость рыскания, рад/с
        
        Returns:
            derivative: вектор производных (12 компонент)
        """
        # Распаковка состояния
        phi, theta, psi = state[3], state[4], state[5]
        vx, vy, vz = state[6], state[7], state[8]
        wx, wy, wz = state[9], state[10], state[11]
        
        # ===== НОРМАЛИЗАЦИЯ УГЛОВ =====
        phi = np.arctan2(np.sin(phi), np.cos(phi))
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        psi = np.arctan2(np.sin(psi), np.cos(psi))
        
        # ===== ОГРАНИЧЕНИЕ УГЛОВ (предотвращает опрокидывание) =====
        max_angle = np.radians(50)  # 50 градусов максимум
        phi = np.clip(phi, -max_angle, max_angle)
        theta = np.clip(theta, -max_angle, max_angle)
        
        # Тригонометрические функции
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)
        
        # Защита от сингулярности при cos_theta → 0
        cos_theta_safe = cos_theta
        if abs(cos_theta_safe) < self.p.epsilon_angle:
            cos_theta_safe = self.p.epsilon_angle * np.sign(cos_theta_safe)
        
        tan_theta = np.tan(theta)
        if abs(tan_theta) > 100.0:
            tan_theta = 100.0 * np.sign(tan_theta)
        
        # Сумма квадратов скоростей винтов
        sum_w2 = omega1**2 + omega2**2 + omega3**2 + omega4**2
        
        # === Производные координат ===
        dx = vx
        dy = vy
        dz = vz
        
        # === Производные углов Эйлера ===
        dphi = wx + wy * sin_phi * tan_theta + wz * cos_phi * tan_theta
        dtheta = wy * cos_phi - wz * sin_phi
        dpsi = wy * (sin_phi / cos_theta_safe) + wz * (cos_phi / cos_theta_safe)
        
        # ===== СОПРОТИВЛЕНИЕ ВОЗДУХА =====
        air_density = 1.225  # кг/м³
        Cd = 0.5  # коэффициент лобового сопротивления
        A = 0.05  # площадь миделя, м²
        
        # Квадратичное сопротивление
        drag_force_x = 0.5 * air_density * Cd * A * abs(vx) * vx
        drag_force_y = 0.5 * air_density * Cd * A * abs(vy) * vy
        drag_force_z = 0.5 * air_density * Cd * A * abs(vz) * vz
        
        # Линейное сопротивление (вязкость)
        linear_drag_x = self.linear_drag * vx
        linear_drag_y = self.linear_drag * vy
        linear_drag_z = self.linear_drag * vz
        
        # === Линейные ускорения ===
        term_vx = -cos_phi * sin_theta * cos_psi - sin_phi * sin_psi
        dvx = (self.a1 * term_vx * sum_w2 
               - drag_force_x / self.p.m 
               - linear_drag_x / self.p.m)
        
        term_vy = -cos_phi * sin_theta * sin_psi + sin_phi * cos_psi
        dvy = (self.a1 * term_vy * sum_w2 
               - drag_force_y / self.p.m 
               - linear_drag_y / self.p.m)
        
        dvz = (self.a2 + self.a1 * cos_phi * cos_theta * sum_w2 
               - drag_force_z / self.p.m 
               - linear_drag_z / self.p.m)
        
        # ===== УГЛОВЫЕ УСКОРЕНИЯ С ДЕМПФИРОВАНИЕМ =====
        dwx = ((self.a3 * wy * wz 
                - self.a4 * wy * (omega1 - omega2 + omega3 - omega4)
                + self.a5 * (omega2**2 - omega4**2)) * self.stability_factor 
               - self.damping_roll * wx)
        
        dwy = ((self.a6 * wx * wz
                - self.a4 * wx * (omega1 - omega2 + omega3 - omega4)
                + self.a5 * (omega1**2 - omega3**2)) * self.stability_factor 
               - self.damping_pitch * wy)
        
        dwz = ((self.a7 * wx * wy 
                - self.a8 * (omega1**2 - omega2**2 + omega3**2 - omega4**2)) * self.stability_factor 
               - self.damping_yaw * wz)
        
        # Сборка вектора производных
        derivative = np.array([
            dx, dy, dz,
            dphi, dtheta, dpsi,
            dvx, dvy, dvz,
            dwx, dwy, dwz
        ])
        
        return derivative
    
    def euler_step(self, state: np.ndarray, 
                   omega1: float, omega2: float, omega3: float, omega4: float,
                   dt: float) -> np.ndarray:
        """Интегрирование методом Эйлера."""
        deriv = self.compute_accelerations(state, omega1, omega2, omega3, omega4)
        new_state = state + deriv * dt
        
        # Обновляем углы в state после интегрирования
        new_state[3] = np.arctan2(np.sin(new_state[3]), np.cos(new_state[3]))
        new_state[4] = np.arctan2(np.sin(new_state[4]), np.cos(new_state[4]))
        new_state[5] = np.arctan2(np.sin(new_state[5]), np.cos(new_state[5]))
        
        return new_state
    
    def rk4_step(self, state: np.ndarray,
                 omega1: float, omega2: float, omega3: float, omega4: float,
                 dt: float) -> np.ndarray:
        """Интегрирование методом Рунге-Кутты 4-го порядка."""
        def f(s):
            return self.compute_accelerations(s, omega1, omega2, omega3, omega4)
        
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        
        new_state = state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        
        # Нормализация углов после интегрирования
        new_state[3] = np.arctan2(np.sin(new_state[3]), np.cos(new_state[3]))
        new_state[4] = np.arctan2(np.sin(new_state[4]), np.cos(new_state[4]))
        new_state[5] = np.arctan2(np.sin(new_state[5]), np.cos(new_state[5]))
        
        return new_state
    
    def apply_control_with_delay(self, current_omega: np.ndarray,
                                  target_omega: np.ndarray,
                                  delay_time: float,
                                  dt: float) -> np.ndarray:
        """
        Учет инерции управляющих воздействий (стр. 10-11).
        Линейная интерполяция для плавного перехода.
        """
        if delay_time <= 0.0:
            return target_omega
        
        alpha = min(1.0, dt / delay_time)
        return current_omega + alpha * (target_omega - current_omega)
    
    def check_stability(self, state: np.ndarray) -> bool:
        """
        Проверка стабильности дрона (стр. 6).
        Возвращает False если дрон "свалился".
        """
        phi = abs(state[3])
        theta = abs(state[4])
        
        if phi > self.max_roll_pitch_rad or theta > self.max_roll_pitch_rad:
            return False
        return True
    
    # === Методы для интеграции с другими модулями ===
    
    def get_position(self, state: np.ndarray) -> Tuple[float, float, float]:
        """Возвращает (x, y, z) для визуализации."""
        return state[0], state[1], state[2]
    
    def get_attitude(self, state: np.ndarray) -> Tuple[float, float, float]:
        """Возвращает (φ, θ, ψ) в градусах."""
        return np.degrees(state[3]), np.degrees(state[4]), np.degrees(state[5])
    
    def get_velocities(self, state: np.ndarray) -> Tuple[float, float, float]:
        """Возвращает (vx, vy, vz)."""
        return state[6], state[7], state[8]
    
    def get_angular_velocities(self, state: np.ndarray) -> Tuple[float, float, float]:
        """Возвращает (ωx, ωy, ωz)."""
        return state[9], state[10], state[11]
    
    def reset_state(self) -> np.ndarray:
        """Сброс в начальное положение."""
        return np.zeros(12)
    
    def set_max_roll_pitch(self, max_degrees: float) -> None:
        """Установка максимальных углов крена и тангажа."""
        self.max_roll_pitch_deg = max_degrees
        self.max_roll_pitch_rad = np.radians(max_degrees)