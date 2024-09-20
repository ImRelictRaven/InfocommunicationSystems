import numpy as np
import matplotlib.pyplot as plt

# Опорная частота
fs = 17 * 5  # 85 Hz
t = np.linspace(0, 1, 900)  # временная шкала от 0 до 1 секунды, 1000 точек
n = len(t)
m_single = 0.5
m_double = 0.8

# Функция для преобразования частоты в угловую частоту
def f2w(f):
    return 2 * np.pi * f

# Функция для демодуляции AM-сигнала
def demodulate_am_signal(signal, f, t):
    carrier = np.cos(2 * np.pi * f * t)
    demodulated = signal * carrier
    return np.abs(np.fft.ifft(np.fft.fft(demodulated) * (np.abs(np.fft.fftfreq(len(t), t[1] - t[0])) < 2 * f)))



# Генерация первичных сигналов
sig1 = 1.0 * np.cos(f2w(fs) * t)
sig2 = 2.0 * np.cos(f2w(fs) * t)
sig3 = 1.5 * np.cos(f2w(fs) * t)

# Генерация несущих сигналов
fn = fs
sam1 = np.cos(f2w(fn) * t)
sam2 = np.cos(f2w(fn) * t)
sam3 = np.cos(f2w(fn) * t)

# Амплитудная модуляция для случая с одной гармоникой
sam1_single = sam1.copy()
sam2_single = sam2.copy()
sam3_single = sam3.copy()

for i in range(n):
    sam1_single[i] = sam1_single[i] * (1 + m_single * sig1[i] / 2.0)
    sam2_single[i] = sam2_single[i] * (1 + m_single * sig2[i] / 2.0)
    sam3_single[i] = sam3_single[i] * (1 + m_single * sig3[i] / 2.0)

# Амплитудная модуляция для случая с двумя гармониками
sig1_double = 1.0 * np.cos(f2w(fs) * t) + 0.8 * np.cos(f2w(2 * fs) * t)
sig2_double = 2.0 * np.cos(f2w(fs) * t) + 0.8 * np.cos(f2w(2 * fs) * t)
sig3_double = 1.5 * np.cos(f2w(fs) * t) + 0.8 * np.cos(f2w(2 * fs) * t)

sam1_double = sam1.copy()
sam2_double = sam2.copy()
sam3_double = sam3.copy()

for i in range(n):
    sam1_double[i] = sam1_double[i] * (1 + m_double * sig1_double[i] / 2.0)
    sam2_double[i] = sam2_double[i] * (1 + m_double * sig2_double[i] / 2.0)
    sam3_double[i] = sam3_double[i] * (1 + m_double * sig3_double[i] / 2.0)

# Суммирование сигналов для передачи по одному каналу
combined_signal_single = sam1_single + sam2_single + sam3_single
combined_signal_double = sam1_double + sam2_double + sam3_double


# Демодуляция сигналов с одной гармоникой
demodulated_signals_single = []
for f in range(2):
    demodulated_signal = demodulate_am_signal(combined_signal_single, fn, t)
    demodulated_signals_single.append(demodulated_signal)

# Демодуляция сигналов с двумя гармониками
demodulated_signals_double = []
for f in range(2):
    demodulated_signal = demodulate_am_signal(combined_signal_double, fn, t)
    demodulated_signals_double.append(demodulated_signal)

# Визуализация сигналов
plt.figure(figsize=(24, 16))

# Визуализация исходных сигналов с одной гармоникой
plt.suptitle('AM-сигналы с одиночной и двойной гармоникой', fontsize=16)

for i, signal in enumerate([sig1, sig2, sig3]):
    plt.subplot(6, 3, i + 1)
    plt.plot(t, signal)
    plt.title(f'Исходный сигнал {i+1} (одиночная гармоника)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

# Визуализация суммарного сигнала с одной гармоникой
plt.subplot(6, 3, 4)
plt.plot(t, combined_signal_single)
plt.title('Комбинированный сигнал (одиночная гармоника)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Визуализация демодулированных сигналов с одной гармоникой
for i, signal in enumerate(demodulated_signals_single):
    plt.subplot(6, 3, 5 + i)
    plt.plot(t, signal)
    plt.title(f'Демодулированный сигнал {i+1} (одиночная гармоника)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

# Визуализация исходных сигналов с двумя гармониками
for i, signal in enumerate([sig1_double, sig2_double, sig3_double]):
    plt.subplot(6, 3, 9 + i)
    plt.plot(t, signal)
    plt.title(f'Исходный сигнал {i+1} (двойная гармоника)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

# Визуализация суммарного сигнала с двумя гармониками
plt.subplot(6, 3, 13)
plt.plot(t, combined_signal_double)
plt.title('Комбинированный сигнал (двойная гармоника)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Визуализация демодулированных сигналов с двумя гармониками
for i, signal in enumerate(demodulated_signals_double):
    plt.subplot(6, 3, 14 + i)
    plt.plot(t, signal)
    plt.title(f'Демодулированный сигнал {i+1} (двойная гармоника)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
