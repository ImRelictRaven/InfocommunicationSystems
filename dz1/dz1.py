import numpy as np
from math import *
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

#ВЧ импульс
def hf_pulse(t):
        ret = 0.0
        if (t >= 0 and t <= 0.5*Tc):
                ret = np.cos(2.0*np.pi*fc*t)
        return ret

#Производная ВЧ импульса
def d_hf_pulse(t):
        ret = 0.0
        if (t >= 0 and t <= 0.5*Tc):
                ret = -2.0*np.pi*fc*np.sin(2.0*np.pi*fc*t)
        return ret

#НЧ импульс
def lf_pulse(t):
        ret = 1.0 if (t >= 0 and t <= 0.5*Tc) else 0.0
        if (t >= 0 and t <= (0+0.005*Tc)):
                if (0.005*Tc > 0.0):
                        ret *= 0.5*(1.0-np.cos(np.pi*(t-0)/0.005*Tc))
        if (t >= (0.5*Tc-0.005*Tc) and t <= 0.5*Tc):
                if (0.005*Tc > 0.0):
                        ret *= 0.5*(1.0-np.cos(np.pi*(0.5*Tc-t)/0.005*Tc))
        return ret

#Производная НЧ импульса
def d_lf_pulse(t):
        ret = 0.0
        if (t >= 0 and t <= (0+0.005*Tc)):
                if (0.005*Tc > 0.0):
                        ret = 0.5*np.sin(np.pi*(t-0)/0.005*Tc)*np.pi/0.005*Tc
        if (t >= (0.5*Tc-0.005*Tc) and t <= 0.5*Tc):
                if (0.005*Tc > 0.0):
                        ret = -0.5*np.sin(np.pi*(0.5*Tc-t)/0.005*Tc)*np.pi/0.005*Tc
        return ret

#Широкополосный импульс
def wb_pulse(t):
        freq = (1*fh + 1.0*fl) * 0.5
        dt = 1 / (1*fh-1.0*fl)
        return np.exp((0.5*Tc-t)**2/dt**2*0.5)*np.sin(2.0*np.pi*freq*t)

#Производная широкополосного импульса
def d_wb_pulse(t):
        freq = (1.5*fh + 0.5*fl) * 0.5
        dt = 1 / (1.5*fh-0.5*fl)
        return 2.0*np.pi*freq*t*np.exp(-(0.5*Tc-t)**2/dt**2*0.5)*np.cos(2.0*np.pi*freq*t)+-2.0*t/(2.0*dt**2)*np.exp(-(0.5*Tc-t)**2/dt**2*0.5)*np.sin(2.0*np.pi*freq*t)

#Задание производной сигнала возбуждения ЛП
def d_signal(t): 
    return  d_wb_pulse(t) # функция изменяется относительно nsig



#Перевод частоты в циклическую
def f2w(f):
    return 2.0*pi*f
def Z1(f, C1):   
    return 2.0/(1j*f2w(f)*C1)
def Z2 (f, C2):
    return 1.0/(1j*f2w(f)*C2)
def Z3(f, L):
    return 1.0j*f2w(f)*L



#Постоянная распространения отдельной ячейки
def Gam(f, L, C1, C2):
    ZY = (Z2(f, C2)+Z3(f, L))/Z1(f, C1)
    return 2.0 * np.arcsinh(np.sqrt(ZY))

#Характеристическое сопротивление отдельной ячейки
def Zw(f, L, C1, C2):
    return np.sqrt((Z1(f, C1)**2*(Z2(f, C2)+Z3(f, L)))/(2*Z1(f, C1)+Z2(f, C2)+Z3(f, L)))

global fc, L, C1, C2, G, aU, dU, aV, dV, time, A0, AN, K0, KN, Tc


Tc = 2 #float(input('Временной интервал '))
N = 17
fl = N #float(input('Нижняя граничная частота ЛП '))
fh = 10*(N+1) #float(input('Верхняя граничная частота ЛП '))
f0 = (fl + fh) * 0.5
Z0 = 10*N #float(input('Характеристическое сопротивление одного звена ЛП на частоте '+str(f0)+' '))
Nc = 20
fc = 0.95*fh #f0 #0.95*fh #f0
L = (sqrt(Z0**2*f2w(f0)**2*(2*f2w(fh)**2-f2w(fl)**2-f2w(f0)**2)/
    ((f2w(fh)**2-f2w(fl)**2)**2*(f2w(f0)**2-f2w(fl)**2))))
C1 = 1.0 / L / (f2w(fh)**2 - f2w(fl)**2)
C2 = 1.0 / (f2w(fl)**2 * L)
#G = 0

print('Параметры отдельной ячейки ЛП:')
print('C1 = {0: f}\nC2 = {1: f}\nL = {2: f}'.format(C1, C2, L))

npp = 15            #Количество точек на период гармонического сигнала
dt = 1/(fc*npp)     #Шаг по времени
num = int(Tc / dt)  #Количество временных отсчетов

freq = np.linspace(0.8*fl, fh*1.2, num)

Gama = Gam(freq, L, C1, C2)
Zw = Zw(freq, L, C1, C2)
dF = (Gam(freq+0.1, L, C1, C2).imag-Gam(freq-0.1, L, C1, C2).imag)/0.2

A0 = 1 #Амплитуда сигнала слева
AN = 0 #Амплитуда сигнала справа
K0 = KN = 1 #Коэффициенты при нагрузочных сопротивлениях

#Количество итераций для решения уравнений возбуждения 
dpp = 20
print('dpp = {0: d}'.format(dpp))

aU = [0] * Nc     #Массив напряжений на емкости C2
dU = [0] * Nc     #Массив производных напряжений на емкости C2
aV = [0] * (Nc+1) #Массив напряжений на емкости C1
dV = [0] * (Nc+1) #Массив производных напряжений на емкости C1

Vinp = [0] * num  #Массив входных напряжений
Vout = [0] * num  #Массив выходных напряжений
time = [0] * num  #Массив временных отсчетов

Vs = [0] * npp    #Массив напряжений на C1 вдоль ЛП на одном периоде сигнала
for i in range(npp): Vs[i] = [0] * (Nc+1)

#Решение уравнений возбуждения ЛП
for it in range(num):        
    time[it] = dt * it
    for i in range(dpp):
        dV[0] += (1.0/(L*C1)*(aV[1]-aV[0]+aU[0])+1.0/(Z0*K0*C1)*(A0*d_signal(time[it])-dV[0]))*dt/dpp
        for ic in range (Nc):
            dU[ic] += (1.0/(L*C2)*(aV[ic]-aV[ic+1]-aU[ic])-G/C2*dU[ic])*dt/dpp
            if ic == 0: continue
            dV[ic] += (0.5/(L*C1)*(aV[ic-1]-2.0*aV[ic]+aV[ic+1]+aU[ic]-aU[ic-1]))*dt/dpp
        dV[Nc] += (1.0/(L*C1)*(aV[Nc-1]-aV[Nc]-aU[Nc-1])+1.0/(Z0*KN*C1)*(AN*d_signal(time[it])-dV[Nc]))*dt/dpp

        for ic in range(Nc):
            aV[ic] += dV[ic]*dt/dpp
            aU[ic] += dU[ic]*dt/dpp
        aV[Nc] += dV[Nc]*dt/dpp
    
    if num-it <= npp:
        for ic in range(Nc+1):
            Vs[it-(num-npp)][ic] = aV[ic]

    Vinp[it] = aV[0]
    Vout[it] = aV[Nc]
    if it % 100 == 0:
        print('{0: 7.3f} {1: 7.3f} {2: 7.3f} '.format(time[it], Vinp[it], Vout[it]))

#Расчет спектра входного и выходного сигалов
spectr_inp = np.fft.fft(Vinp)
spectr_out = np.fft.fft(Vout)
fft_freq = np.fft.fftfreq(num, Tc/num)



plt.figure()
plt.plot(time, Vinp, time, Vout)
plt.title("Uвх(t),Uвых(t)")
plt.show()
plt.figure()
sp_inp = np.hypot(spectr_inp.real, spectr_inp.imag)/num*2
sp_out = np.hypot(spectr_out.real, spectr_out.imag)/num*2
plt.plot(fft_freq[0:num//2], sp_inp[0:num//2], label='$V_{inp}$')
plt.plot(fft_freq[0:num//2], sp_out[0:num//2], label='$V_{out}$')
plt.title("Спектры входных и выходных напряжений(Uвх(f),Uвых(f))")
plt.legend(loc='best')
plt.show()



plt.figure()
plt.plot(freq, abs(Zw), label='$|Z_0|(f)$')
plt.plot(freq, Zw.real, label='$Re(Z_0)(f)$')
plt.plot(freq, Zw.imag, label='$Im(Z_0)(f)$')
plt.title("Зависимость волнового напряжения Z0(f)")
plt.vlines(fc, 0, Z0, color='tab:olive', linestyles='dashdot', lw=1)
plt.hlines(Z0, freq[0], fc, color='tab:olive', linestyles='dashdot', lw=1)
plt.legend(loc='best')
plt.show()

plt.figure()
plt.title("Зависимость фазового сдвига")
plt.plot(freq, Gama.imag, color='tab:orange', label=r'$\phi(f)$')
plt.tick_params(axis='y', labelcolor='tab:orange')
plt.legend(loc='upper left')
plt.show()


plt.figure()
plt.title("Распределение напряжений по ячейкам вдоль линии передачи в моменты времени")
cells = np.linspace(0, Nc, Nc+1)
z_spl = np.linspace(0, Nc, (Nc+1)*10)
for i in range(npp):
    spl = make_interp_spline(cells, Vs[i], k=3)
    plt.plot(z_spl, spl(z_spl), label="t = {0: .3f} Ñ ".format(time[num-npp+i]), lw=1)
plt.legend(loc='best')
plt.show()
