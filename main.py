# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Process, Manager
import sys


class Aircraft:
    def __init__(self, H: float = 5000):
        self.x = self.y = 0.
        self.z = H

    @property
    def coords(self):
        return np.asarray([self.x, self.y, self.z])


class Reflector:
    def __init__(self, D: float = 5000, L0: float = 500, n: int = 1000, z_loc: float = 4., z_scale: float = 2.):
        x = np.random.random(n) * L0 - L0 / 2.
        # x = np.linspace(-L0 / 2., L0 / 2., n)
        y = np.ones_like(x) * D
        z = np.random.normal(loc=z_loc, scale=z_scale, size=n)
        self.coords = np.asarray([x, y, z]).T


class Model:
    def __init__(self, aircraft: Aircraft, reflector: Reflector):
        self.aircraft = aircraft
        self.reflector = reflector

    def go(self, start: float = 0., stop: float = 10000., dx: float = 1., wavelength: float = 0.23, A0: float = 1.,
           verbose=True):
        diap = np.arange(start, stop, dx)
        signal = []
        for i, x in enumerate(diap):
            self.aircraft.x = x
            l = np.linalg.norm(self.reflector.coords - self.aircraft.coords, axis=-1)
            k = 2 * np.pi / wavelength
            d = (np.power(4 * np.pi, 2) * np.power(l, 4))
            # d = 1.
            sig = np.sum(A0 * np.exp(1j * k * 2 * l) / d)
            signal.append(sig)
            if verbose:
                print('\r{:.2f}%'.format(i / len(diap) * 100), end='   ', flush=True)
        if verbose:
            print()
        return diap, signal


def process(DIAP_, AMPL_, PHI_, k_):
    np.random.seed(np.random.randint(0, 2**32 - 1))
    plane = Aircraft(H=H)
    surface = Reflector(D=D, L0=L0, n=n, z_loc=h_mean, z_scale=np.sqrt(h_disp))
    model = Model(plane, surface)
    diap, signal = model.go(start=start, dx=dx, stop=stop, wavelength=wavelength, A0=A0, verbose=False)
    DIAP_.append(diap)

    ampl, phi = np.absolute(signal), np.angle(signal)
    AMPL_.append(ampl)
    PHI_.append(phi)

    k_.value += 1
    if N-1:
        print('\rTotal: {:.2f}%'.format(k_.value / (N-1) * 100.), end='     ', flush=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    H = 5000
    D = 5000
    L0 = 500
    n = 100000
    # n = 1000000
    h_mean = 4
    h_disp = 2
    start = -10
    stop = 10
    # wavelength = 0.03
    wavelength = 0.23
    # wavelength = 0.7
    # dx = wavelength / 10
    dx = 0.1
    A0 = 1
    N = 1
    n_workers = 8

    print()
    with Manager() as manager:
        AMPL = manager.list()
        PHI = manager.list()
        DIAP = manager.list()
        k = manager.Value('i', 0)

        processes = []
        for _ in range(N):
            p = Process(target=process, args=(DIAP, AMPL, PHI, k))
            processes.append(p)

        for i in range(0, len(processes), n_workers):
            for j in range(i, i + n_workers):
                if j < len(processes):
                    processes[j].start()
            for j in range(i, i + n_workers):
                if j < len(processes):
                    processes[j].join()

        DIAP = list(DIAP)
        AMPL = list(AMPL)
        PHI = list(PHI)

    ampl = np.asarray(AMPL).mean(axis=0)
    phi = np.asarray(PHI).mean(axis=0)
    diap = np.asarray(DIAP).mean(axis=0)

    phi_corr = np.correlate(phi, phi, mode='full')
    amp_corr = np.correlate(ampl, ampl, mode='full')
    phi_corr /= np.max(np.abs(phi_corr))
    amp_corr /= np.max(np.abs(amp_corr))

    fig, ax = plt.subplots()
    ax.set_xlabel(r'Перемещение в $\lambda$')
    ax.set_ylabel('Амплитуда')
    ax.plot(diap / wavelength, ampl, label='Амплитуда', color='darkorange')
    ax.legend(loc='upper left', frameon=False)
    ax = ax.twinx()
    ax.set_ylabel('Фаза')
    ax.set_ylim((-np.pi, np.pi))
    ax.plot(diap / wavelength, phi, label='Фаза', color='darkblue')
    ax.legend(loc='upper right', frameon=False)
    plt.savefig('timeseries.png', dpi=300)

    plt.figure()
    plt.xlabel(r'Перемещение в $\lambda$')
    plt.ylabel('Значения')
    x = np.linspace(-stop / wavelength, stop / wavelength, len(phi_corr))
    plt.plot(x, phi_corr, label='Автокорреляция фазы', color='black', linestyle='-')
    plt.plot(x, amp_corr, label='Автокорреляция амплитуды', color='black', linestyle='-.')
    plt.hlines(0.5, -stop / wavelength, stop / wavelength, color='black', linestyle='--')
    plt.legend(loc='best', frameon=False)
    plt.savefig('autocorr.png', dpi=300)
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
