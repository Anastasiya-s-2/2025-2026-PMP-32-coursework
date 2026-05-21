import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def sird_optimal_control():
    T_max = 100
    N = 1000
    beta = 0.35
    gamma = 0.12
    mu = 0.03
    S0 = 900
    I0 = 100
    A1 = 10.0  
    A2 = 10.0   
    A3 = 10.0 
    A4 = 0.5
    w1=10000
    w2=100000
    w3=100000
    t = np.linspace(0, T_max, 1000)
    dt = t[1] - t[0]
    def sird_no_control(t, y):
        S, I, R, D = y
        dS = -beta * S * I / N
        dI = beta * S * I / N - (gamma + mu) * I
        dR = gamma * I
        dD = mu * I
        return [dS, dI, dR, dD]
    
    sol_free = solve_ivp(sird_no_control, [0, T_max], [S0, I0, 0, 0],
                         t_eval=t, method='RK45', rtol=1e-8, atol=1e-10)

    S_free, I_free, R_free, D_free = sol_free.y
    print(f"Пік інфікованих:      {np.max(I_free):.1f}")
    print(f"Сприйнятливі в кінці: {S_free[-1]:.1f}")
    print(f"Одужалі в кінці:      {R_free[-1]:.1f}")
    print(f"Померлі в кінці:      {D_free[-1]:.1f}\n")
    v1 = np.zeros_like(t)
    v2 = np.zeros_like(t)
    v3 = np.zeros_like(t)
    
    for it in range(200):
        v1_old = v1.copy()
        v2_old = v2.copy()
        v3_old = v3.copy()
        def sird_forward(t_val, y, v1f, v2f, v3f):
            S, I, R, D = y
            v1t = v1f(t_val)
            v2t = v2f(t_val)
            v3t = v3f(t_val)
            dS = -(1 - v2t) * beta * S * I / N - v1t * S
            dI =  (1 - v2t) * beta * S * I / N - (gamma + mu + v3t) * I
            dR = gamma * I + v1t * S + v3t * I
            dD = mu * I
            return [dS, dI, dR, dD]
        v1f = lambda tt: np.interp(tt, t, v1)
        v2f = lambda tt: np.interp(tt, t, v2)
        v3f = lambda tt: np.interp(tt, t, v3)
        sol = solve_ivp(sird_forward, [0, T_max], [S0, I0, 0, 0],
                        args=(v1f, v2f, v3f), method='RK45',
                        t_eval=t, rtol=1e-8, atol=1e-10)
        S, I, R, D = sol.y
        p1 = np.zeros_like(t)
        p2 = np.zeros_like(t)
        p3 = np.zeros_like(t)
        p4 = np.zeros_like(t)
        
        for i in range(len(t)-2, -1, -1):
            v1t = v1[i+1]
            v2t = v2[i+1]
            v3t = v3[i+1]
            St = S[i+1]
            It = I[i+1]

            dp1 = -A4 + p1[i+1]*((1-v2t)*beta*It/N + v1t) - p2[i+1]*(1-v2t)*beta*It/N - p3[i+1]*v1t
            dp2 = -A1 + (p1[i+1]-p2[i+1])*(1-v2t)*beta*St/N + p2[i+1]*(gamma + mu + v3t) \
                  - p3[i+1]*(gamma + v3t) - p4[i+1]*mu

            p1[i] = p1[i+1] - dt * dp1
            p2[i] = p2[i+1] - dt * dp2
            p3[i] = p3[i+1] - dt * A3
            p4[i] = p4[i+1] + dt * A2
        v1_new = np.clip((p1 - p3) * S / w1, 0, 1)
        v2_new = np.clip(-(p1 - p2) * beta * S * I / (N * w2), 0, 1)
        v3_new = np.clip((p2 - p3) * I / w3, 0, 1)
        alpha = 0.6
        
        v1 = alpha * v1_new + (1 - alpha) * v1_old
        v2 = alpha * v2_new + (1 - alpha) * v2_old
        v3 = alpha * v3_new + (1 - alpha) * v3_old
    print(f"Пік інфікованих:      {np.max(I):.1f}")
    print(f"Сприйнятливі в кінці: {S[-1]:.1f}")
    print(f"Одужалі в кінці:      {R[-1]:.1f}")
    print(f"Померлі в кінці:      {D[-1]:.1f}")
    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(t, S, 'b', label='S — сприйнятливі')
    ax1.plot(t, I, 'r', label='I — інфіковані')
    ax1.plot(t, R, 'g', label='R — одужалі')
    ax1.plot(t, D, 'k', label='D — померлі')
    ax1.set_title('З оптимальним керуванням')
    ax1.legend()
    ax1.grid(True)

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(t, S_free, 'b', label='S — сприйнятливі')
    ax2.plot(t, I_free, 'r', label='I — інфіковані')
    ax2.plot(t, R_free, 'g', label='R — одужалі')
    ax2.plot(t, D_free, 'k', label='D — померлі')
    ax2.set_title('Без керування')
    ax2.legend()
    ax2.grid(True)
    ax3 = plt.subplot(2, 1, 2)
    ax3.plot(t, v1, 'b', label='v₁')
    ax3.plot(t, v2, 'orange', label='v₂')
    ax3.plot(t, v3, 'g', label='v₃ ')
    ax3.set_title('Оптимальні керуючі впливи')
    ax3.set_xlabel('Час (дні)')
    ax3.set_ylabel('Інтенсивність керуючих впливів (0–1)')
    ax3.legend()
    ax3.grid(True)
    plt.tight_layout()
    plt.show()

sird_optimal_control()
