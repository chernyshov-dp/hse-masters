import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.constants import hbar

# Параметры системы
L = 10  # Размер области
N = 500  # Количество точек на сетке
dx = L / N  # Шаг по координате
dt = 0.01 # Шаг по времени
x = np.linspace(0, L, N)  # Создание сетки по координате

# Начальное состояние волнового пакета (осциллирующий гауссиан)
x0 = 3.0
k0 = 3.0
sigma0 = 1.0
psi0 = np.exp(-(x - x0)**2 / (2 * sigma0**2)) * np.exp(1j * k0 * x)

# Произвольный потенциал (в данном случае, просто яма)
V = np.zeros_like(x)
V[(x > 7) & (x < 8)] = 10

fig, ax = plt.subplots()
line, = ax.plot(x, np.abs(psi0)**2)

def update(frame):
    global psi0
    psi0 = psi0 * np.exp(-1j * V * dt / (2 * hbar))
    psi0 = np.fft.fft(psi0)
    psi0 = psi0 * np.exp(-1j * hbar * (2 * np.pi * np.fft.fftfreq(N, dx))**2 * dt / (2 * hbar))
    psi0 = np.fft.ifft(psi0)
    psi0 = psi0 * np.exp(-1j * V * dt / (2 * hbar))
    
    line.set_ydata(np.abs(psi0)**2)
    return line,


ani = FuncAnimation(fig, update, frames=500, interval=20, blit=True)

ax.plot(x, V, 'k--', label='Potential')
ax.set_title('Dynamics of an Oscillating Wave Packet in a Potential Well')
ax.set_xlabel('Position')
ax.set_ylabel('Probability Density')
ax.legend()

ani.save('wave_packet_animation.gif', writer='pillow')

plt.show()
