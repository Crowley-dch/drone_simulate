"""
controller.py — Модуль управления и случайных циклов (Задание 2)

Реализует:
- Генерацию случайных управляющих последовательностей (формулы 4–6)
- Перевод напряжений Up в угловые скорости ω (формула 3)
- Контроль высоты и направления полёта
- Главный цикл симуляции по алгоритму рис. 5

Интеграция с другими модулями:
- physics_callback(state, omega, dt) -> new_state   (модуль physics.py, Человек 1)
- visualization_callback(state)                     (модуль camera.py, Человек 3)
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


# ─────────────────────────────────────────────
# Параметры по умолчанию (Таблица 1 статьи)
# ─────────────────────────────────────────────
DEFAULT_PARAMS = {
    # Электрические / мотор
    "KV": 920,          # об/мин/В (типовое значение для 1 кг БпЛА)
    "eta": 0.7,         # к.п.д. под нагрузкой (η = 0.6…0.8)
    "V_MIN": 6.0,       # минимальное напряжение на двигателях, В
    "V_MAX": 12.0,      # максимальное напряжение на двигателях, В

    # Полётные ограничения
    "z_min": 5.0,       # минимальная высота полёта, м
    "dt": 0.04,         # шаг интегрирования / кадр, с (40 мс = 24 Гц)

    # Параметры цикла
    "n_cycles": 30,              # количество случайных циклов управления
    "t_cycle_min": 1.0,          # минимальная длительность цикла, с
    "t_cycle_max": 4.0,          # максимальная длительность цикла, с

    # Параметры задержки (нормальное распределение)
    "delay_mean": 0.2,           # математическое ожидание задержки, с
    "delay_std": 0.05,           # среднеквадратическое отклонение, с

    # Стандартный (взлётный) цикл
    "V_hover": 9.0,              # напряжение одинаковое на все 4 мотора при взлёте
    "t_takeoff": 3.0,            # длительность взлётного цикла, с
}


# ─────────────────────────────────────────────
# Вспомогательные структуры данных
# ─────────────────────────────────────────────

@dataclass
class MotorCommand:
    """Команда на один двигатель в рамках цикла управления."""
    U_p: float          # целевое напряжение, В
    t_delay: float      # задержка установления нового режима, с


@dataclass
class ControlCycle:
    """Один цикл управления n (строка матрицы (4))."""
    n: int                              # номер цикла
    motors: List[MotorCommand]          # 4 команды для 4 двигателей
    t_cycle: float                      # продолжительность цикла, с
    is_standard: bool = False           # True → взлётный цикл


@dataclass
class FlightPlan:
    """Матрица циклов управления (формула 4)."""
    cycles: List[ControlCycle] = field(default_factory=list)


# ─────────────────────────────────────────────
# Формула (3): напряжение → угловая скорость
# ─────────────────────────────────────────────

def voltage_to_omega(U_p: float, KV: float, eta: float) -> float:
    """
    Перевод напряжения Up в угловую скорость ω (рад/с).

    Формула (3) из статьи:
        ω = U_p * KV * η * 6.28 / 60

    Параметры
    ----------
    U_p  : напряжение на выходе регулятора двигателя, В
    KV   : об/мин/В (паспортный показатель мотора)
    eta  : η — коэффициент к.п.д. под нагрузкой (0.6…0.8)

    Возвращает
    ----------
    ω : угловая скорость пропеллера, рад/с
    """
    return U_p * KV * eta * (2.0 * np.pi) / 60.0


def voltages_to_omegas(
    U_p_list: List[float],
    KV: float,
    eta: float,
) -> np.ndarray:
    """Перевод списка из 4 напряжений в вектор угловых скоростей."""
    return np.array([voltage_to_omega(u, KV, eta) for u in U_p_list])


# ─────────────────────────────────────────────
# Генерация матрицы циклов управления (форм. 4–6)
# ─────────────────────────────────────────────

def generate_random_flight_plan(
    num_cycles: int,
    V_MIN: float,
    V_MAX: float,
    V_hover: float,
    t_takeoff: float,
    t_cycle_min: float,
    t_cycle_max: float,
    delay_mean: float,
    delay_std: float,
) -> FlightPlan:
    """
    Генерация матрицы случайных циклов управления (формулы 4–6).

    Цикл n=0 всегда «стандартный» (вертикальный набор высоты):
        все 4 канала получают одинаковое напряжение V_hover.
    Циклы n=1…N — случайные:
        Up ∈ [V_MIN, V_MAX] для каждого канала независимо,
        задержка t_з ~ N(delay_mean, delay_std²).

    Возвращает
    ----------
    FlightPlan с (num_cycles + 1) циклами (0 взлётный + num_cycles случайных).
    """
    rng = np.random.default_rng()
    plan = FlightPlan()

    # ── Цикл 0: стандартный взлётный ──────────────────────────────────────
    std_motors = [
        MotorCommand(
            U_p=V_hover,
            t_delay=max(0.0, rng.normal(delay_mean, delay_std)),
        )
        for _ in range(4)
    ]
    plan.cycles.append(
        ControlCycle(n=0, motors=std_motors, t_cycle=t_takeoff, is_standard=True)
    )

    # ── Циклы 1…N: случайные ──────────────────────────────────────────────
    for n in range(1, num_cycles + 1):
        motors = []
        for _ in range(4):
            U_p = rng.uniform(V_MIN, V_MAX)                          # Up ∈ [V_MIN, V_MAX]
            t_delay = max(0.0, rng.normal(delay_mean, delay_std))    # t_з ~ N(0.2, 0.05)
            motors.append(MotorCommand(U_p=U_p, t_delay=t_delay))

        t_cycle = rng.uniform(t_cycle_min, t_cycle_max)
        plan.cycles.append(ControlCycle(n=n, motors=motors, t_cycle=t_cycle))

    return plan


# ─────────────────────────────────────────────
# Вычисление текущего вектора напряжений (форм. 5)
# ─────────────────────────────────────────────

def compute_current_voltages(
    cycle: ControlCycle,
    prev_cycle: Optional[ControlCycle],
    t_local: float,
) -> List[float]:
    """
    Текущие напряжения на 4 каналах в момент t_local внутри цикла.

    Формула (5):
        если t_local < t_delay[i] → берём напряжение предыдущего цикла
        иначе                     → берём напряжение текущего цикла

    Параметры
    ----------
    cycle      : текущий цикл управления
    prev_cycle : предыдущий цикл (None для первого)
    t_local    : время от начала текущего цикла, с

    Возвращает
    ----------
    Список из 4 текущих напряжений.
    """
    voltages = []
    for i in range(4):
        t_delay = cycle.motors[i].t_delay
        if t_local < t_delay:
            # Действует напряжение предыдущего цикла (или 0 в самом начале)
            if prev_cycle is not None:
                voltages.append(prev_cycle.motors[i].U_p)
            else:
                voltages.append(0.0)
        else:
            voltages.append(cycle.motors[i].U_p)
    return voltages


# ─────────────────────────────────────────────
# Стандартный «взлётный» мини-цикл (для аварийного восстановления высоты)
# ─────────────────────────────────────────────

def make_recovery_cycle(
    V_hover: float,
    t_takeoff: float,
    delay_mean: float,
    delay_std: float,
    cycle_index: int = -1,
) -> ControlCycle:
    """
    Создаёт стандартный взлётный цикл для восстановления высоты.
    Используется при z_k < z_min или y_k < 0.
    """
    rng = np.random.default_rng()
    motors = [
        MotorCommand(
            U_p=V_hover,
            t_delay=max(0.0, rng.normal(delay_mean, delay_std)),
        )
        for _ in range(4)
    ]
    return ControlCycle(
        n=cycle_index,
        motors=motors,
        t_cycle=t_takeoff,
        is_standard=True,
    )


# ─────────────────────────────────────────────
# Главный цикл симуляции (алгоритм рис. 5)
# ─────────────────────────────────────────────

def run_simulation(
    flight_plan: FlightPlan,
    physics_callback: Callable,
    visualization_callback: Optional[Callable],
    params: Optional[dict] = None,
    real_time: bool = False,
) -> List[np.ndarray]:
    """
    Запуск имитационной модели полёта БпЛА (алгоритм рис. 5).

    Параметры
    ----------
    flight_plan          : матрица циклов управления (от generate_random_flight_plan)
    physics_callback     : функция update_state(state, omega, dt) -> new_state
                           (Человек 1, модуль physics.py)
    visualization_callback : функция draw(state) или None
                             (Человек 3, модуль camera.py)
    params               : словарь параметров (см. DEFAULT_PARAMS)
    real_time            : если True — добавляет паузу dt между шагами

    Возвращает
    ----------
    Список векторов состояния на каждом шаге (траектория).
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    KV         = params["KV"]
    eta        = params["eta"]
    V_hover    = params["V_hover"]
    t_takeoff  = params["t_takeoff"]
    delay_mean = params["delay_mean"]
    delay_std  = params["delay_std"]
    z_min      = params["z_min"]
    dt         = params["dt"]

    # Начальное состояние БпЛА: все нули (начало координат ССК)
    # state = [x, y, z, phi, theta, psi, vx, vy, vz, wx, wy, wz]
    state = np.zeros(12)

    trajectory: List[np.ndarray] = [state.copy()]

    cycles = flight_plan.cycles
    total_cycles = len(cycles)

    print(f"[Controller] Запуск симуляции: {total_cycles} циклов управления")
    print(f"[Controller] dt={dt*1000:.0f} мс, z_min={z_min} м")

    prev_cycle: Optional[ControlCycle] = None
    n = 0  # индекс текущего цикла

    while n < total_cycles:
        cycle = cycles[n]
        t_local = 0.0
        interrupted = False

        cycle_type = "ВЗЛЁТНЫЙ" if cycle.is_standard else f"СЛУЧАЙНЫЙ #{cycle.n}"
        print(f"\n[Controller] Цикл {n}/{total_cycles-1} [{cycle_type}], "
              f"t_cycle={cycle.t_cycle:.2f} с")

        # ── Внутренний цикл по шагам dt ────────────────────────────────────
        while t_local < cycle.t_cycle:
            # Шаг 6: вычислить текущие напряжения (формула 5)
            voltages = compute_current_voltages(cycle, prev_cycle, t_local)

            # Шаг 8: перевод Up → ω (формула 3)
            omega = voltages_to_omegas(voltages, KV, eta)

            # Шаг 9: решение уравнений (1)–(2) → новые координаты БпЛА
            state = physics_callback(state, omega, dt)
            trajectory.append(state.copy())

            x_k, y_k, z_k = state[0], state[1], state[2]

            # ── Проверка контроля высоты (Шаг 9 алгоритма) ───────────────
            if z_k < z_min:
                print(f"  [!] z_k={z_k:.2f} м < z_min={z_min} м — "
                      f"прерываем цикл, выполняем взлётный цикл")
                interrupted = True
                break

            # ── Проверка направления полёта: y_k >= 0 ─────────────────────
            if y_k < 0:
                print(f"  [!] y_k={y_k:.2f} м < 0 — "
                      f"БпЛА улетел за камеру, выполняем взлётный цикл")
                interrupted = True
                break

            # ── Визуализация (Шаг 8 алгоритма) ────────────────────────────
            if visualization_callback is not None:
                visualization_callback(state)

            # ── Задержка реального времени ─────────────────────────────────
            if real_time:
                time.sleep(dt)

            t_local += dt

        # ── Обработка прерывания ──────────────────────────────────────────
        if interrupted:
            # Вставляем аварийный взлётный цикл перед следующим
            recovery = make_recovery_cycle(
                V_hover, t_takeoff, delay_mean, delay_std, cycle_index=-1
            )
            _run_recovery(
                recovery, prev_cycle, state, trajectory,
                physics_callback, visualization_callback,
                KV, eta, z_min, dt, real_time,
            )
            # После восстановления высоты переходим к СЛЕДУЮЩЕМУ циклу,
            # а не возвращаемся к прерванному (согласно алгоритму шаг 9)
            prev_cycle = recovery
            n += 1
        else:
            prev_cycle = cycle
            n += 1

    x_k, y_k, z_k = state[0], state[1], state[2]
    print(f"\n[Controller] Симуляция завершена. "
          f"Итоговая позиция: x={x_k:.2f}, y={y_k:.2f}, z={z_k:.2f}")
    print(f"[Controller] Всего шагов: {len(trajectory)}, "
          f"время: {len(trajectory)*dt:.1f} с")

    return trajectory


def _run_recovery(
    recovery: ControlCycle,
    prev_cycle: Optional[ControlCycle],
    state: np.ndarray,
    trajectory: List[np.ndarray],
    physics_callback: Callable,
    visualization_callback: Optional[Callable],
    KV: float,
    eta: float,
    z_min: float,
    dt: float,
    real_time: bool,
) -> None:
    """
    Выполняет аварийный взлётный цикл восстановления высоты.
    Изменяет state и trajectory на месте.
    """
    print(f"  [Recovery] Запуск взлётного цикла на {recovery.t_cycle:.1f} с")
    t_local = 0.0
    while t_local < recovery.t_cycle:
        voltages = compute_current_voltages(recovery, prev_cycle, t_local)
        omega = voltages_to_omegas(voltages, KV, eta)
        new_state = physics_callback(state, omega, dt)
        state[:] = new_state
        trajectory.append(state.copy())

        if visualization_callback is not None:
            visualization_callback(state)

        if real_time:
            time.sleep(dt)

        t_local += dt

    print(f"  [Recovery] Высота после восстановления: z={state[2]:.2f} м")


# ─────────────────────────────────────────────
# Удобная точка входа с параметрами из словаря
# ─────────────────────────────────────────────

def create_and_run(
    physics_callback: Callable,
    visualization_callback: Optional[Callable] = None,
    params: Optional[dict] = None,
    real_time: bool = False,
) -> Tuple[FlightPlan, List[np.ndarray]]:
    """
    Вспомогательная функция: создаёт план полёта и запускает симуляцию.

    Параметры
    ----------
    physics_callback       : state, omega, dt -> new_state  (Человек 1)
    visualization_callback : state -> None                  (Человек 3)
    params                 : словарь параметров БпЛА
    real_time              : включить паузы между шагами

    Возвращает
    ----------
    (flight_plan, trajectory)
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    plan = generate_random_flight_plan(
        num_cycles  = params["n_cycles"],
        V_MIN       = params["V_MIN"],
        V_MAX       = params["V_MAX"],
        V_hover     = params["V_hover"],
        t_takeoff   = params["t_takeoff"],
        t_cycle_min = params["t_cycle_min"],
        t_cycle_max = params["t_cycle_max"],
        delay_mean  = params["delay_mean"],
        delay_std   = params["delay_std"],
    )

    trajectory = run_simulation(
        flight_plan            = plan,
        physics_callback       = physics_callback,
        visualization_callback = visualization_callback,
        params                 = params,
        real_time              = real_time,
    )

    return plan, trajectory


# ─────────────────────────────────────────────
# Самостоятельный запуск / демонстрация
# ─────────────────────────────────────────────

def _demo_physics(state: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """
    Заглушка физической модели для демонстрации модуля без physics.py.
    Реализует упрощённую модель: z растёт пропорционально суммарной тяге,
    y медленно растёт (полёт «вперёд»), x зависит от дисбаланса моторов.

    В реальной интеграции замените на physics.integrate(state, omega, dt).
    """
    new_state = state.copy()
    b = 3.13e-5          # коэффициент тяги (из таблицы 1, среднее 10^-5..10^-6)
    g = 9.81
    m = 1.076            # масса, кг

    total_thrust = b * np.sum(omega ** 2)
    az = total_thrust / m - g

    # Обновляем скорости и координаты (метод Эйлера, упрощённо)
    new_state[8] += az * dt                        # vz
    new_state[2] += new_state[8] * dt              # z

    # Небольшое движение «вперёд» по y
    new_state[7] += 0.05 * dt                      # vy
    new_state[1] += new_state[7] * dt              # y

    # Горизонтальный дрейф из-за дисбаланса моторов
    imbalance = (omega[0] + omega[2]) - (omega[1] + omega[3])
    new_state[6] += imbalance * 1e-5 * dt          # vx
    new_state[0] += new_state[6] * dt              # x

    return new_state


if __name__ == "__main__":
    import json

    print("=" * 60)
    print("  controller.py — демонстрационный запуск (заглушка физики)")
    print("=" * 60)

    # Параметры
    params = DEFAULT_PARAMS.copy()
    params["n_cycles"] = 10
    params["t_cycle_min"] = 1.0
    params["t_cycle_max"] = 2.0

    # Генерируем план
    plan = generate_random_flight_plan(
        num_cycles  = params["n_cycles"],
        V_MIN       = params["V_MIN"],
        V_MAX       = params["V_MAX"],
        V_hover     = params["V_hover"],
        t_takeoff   = params["t_takeoff"],
        t_cycle_min = params["t_cycle_min"],
        t_cycle_max = params["t_cycle_max"],
        delay_mean  = params["delay_mean"],
        delay_std   = params["delay_std"],
    )

    print(f"\nСгенерированный план полёта ({len(plan.cycles)} циклов):")
    for cyc in plan.cycles:
        ups = [f"{m.U_p:.2f}В" for m in cyc.motors]
        delays = [f"{m.t_delay:.3f}с" for m in cyc.motors]
        kind = "ВЗЛЁТ" if cyc.is_standard else "СЛУЧ."
        print(f"  n={cyc.n:3d} [{kind}] t={cyc.t_cycle:.2f}с  "
              f"Up=[{', '.join(ups)}]  "
              f"t_з=[{', '.join(delays)}]")

    print("\nЗапуск симуляции...")
    trajectory = run_simulation(
        flight_plan            = plan,
        physics_callback       = _demo_physics,
        visualization_callback = None,
        params                 = params,
        real_time              = False,
    )

    # Вывод ключевых точек траектории
    print(f"\nТраектория (каждые 25 шагов из {len(trajectory)}):")
    print(f"  {'Шаг':>6}  {'x':>8}  {'y':>8}  {'z':>8}  {'vz':>8}")
    for i in range(0, len(trajectory), max(1, len(trajectory) // 25)):
        s = trajectory[i]
        print(f"  {i:>6}  {s[0]:>8.3f}  {s[1]:>8.3f}  {s[2]:>8.3f}  {s[8]:>8.4f}")

    # Краткая статистика
    zs = [s[2] for s in trajectory]
    print(f"\nСтатистика по высоте: "
          f"min={min(zs):.2f} м, max={max(zs):.2f} м, "
          f"финал={zs[-1]:.2f} м")

    print("\nДемонстрация успешно завершена.")
    print("Для реальной интеграции передайте physics.integrate и camera.draw "
          "в run_simulation() или create_and_run().")
