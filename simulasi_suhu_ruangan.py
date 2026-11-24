"""
Modul 6 - Dynamic System Modelling (TA-06)
Studi kasus: Dinamika Suhu Ruangan Ber-AC (model ODE orde satu)

Persamaan:  C dT/dt = UA (T_out - T) + Q_int - Q_max * u(t)
Metode    : Euler eksplisit, dt = 10 s
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------- parameter ----------------
C     = 150000.0   # kapasitas kalor efektif ruangan (J/K)
UA    = 40.0       # konduktansi termal dinding (W/K)
T_out = 32.0       # suhu luar (C)  -> konstanta
Q_int = 200.0      # beban panas internal (W) -> gangguan
Q_max = 600.0      # kapasitas pendinginan AC maksimum (W)
dt    = 10.0       # langkah waktu (s)
t_end = 6 * 3600   # lama simulasi (s)
tau   = C / UA     # konstanta waktu (s)

t = np.arange(0, t_end + dt, dt)
th = t / 3600.0    # waktu dalam jam (untuk plot)

def T_steady(u, Tout=T_out, Qi=Q_int):
    return Tout + (Qi - Q_max * u) / UA

def simulate(u_func, T0, Qi=Q_int):
    T = np.zeros_like(t); u = np.zeros_like(t)
    T[0] = T0; u[0] = u_func(t[0])
    for i in range(1, len(t)):
        dTdt = (UA * (T_out - T[i-1]) + Qi - Q_max * u[i-1]) / C
        T[i] = T[i-1] + dTdt * dt
        u[i] = u_func(t[i])
    return T, u

# ---------------- sinyal kontrol ----------------
t_on = 3600.0  # sinyal aktif pada t = 1 jam
def u_step(tt):   return 0.5 if tt >= t_on else 0.0
def u_impulse(tt):return 1.0 if t_on <= tt < t_on + 300 else 0.0           # pulsa 5 menit
def u_ramp(tt):   return min(max((tt - t_on) / 3600.0 * 0.25, 0.0), 1.0)   # naik 0.25 per jam
def u_sin(tt):    return 0.5 + 0.4*np.sin(2*np.pi*(tt - t_on)/7200.0) if tt >= t_on else 0.0

T0 = T_steady(0.0)   # kondisi awal = steady state tanpa AC (37 C)

if __name__ == "__main__":
    print(f"tau = {tau:.0f} s = {tau/60:.1f} menit")
    for u in (0.0, 0.5, 1.0):
        print(f"steady state u={u:.1f}: T_ss = {T_steady(u):.2f} C")

    # ---- 1. steady state ----
    plt.figure(figsize=(8, 4.5))
    for u, c in ((0.0, "tab:red"), (0.5, "tab:blue"), (1.0, "tab:green")):
        T, _ = simulate(lambda tt, u=u: u, 32.0)
        plt.plot(th, T, color=c, label=f"u = {u:.1f}")
        plt.axhline(T_steady(u), color=c, ls="--", lw=0.8)
    plt.xlabel("Waktu (jam)"); plt.ylabel("Suhu ruangan T (°C)")
    plt.title("Simulasi steady state suhu ruangan (T(0) = 32 °C)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("grafik_steady_state.png", dpi=200); plt.close()

    # ---- 2. verifikasi Euler vs analitik ----
    T_num, _ = simulate(lambda tt: 0.0, 32.0)
    T_ana = T_steady(0.0) + (32.0 - T_steady(0.0)) * np.exp(-t / tau)
    print("galat maks Euler vs analitik:", np.max(np.abs(T_num - T_ana)))

    # ---- 3. respon terhadap sinyal kontrol ----
    skenario = [("Step", u_step), ("Impulse (pulsa 5 menit)", u_impulse),
                ("Ramp", u_ramp), ("Sinusoida", u_sin)]
    fig, ax = plt.subplots(4, 2, figsize=(11, 11))
    for k, (nama, f) in enumerate(skenario):
        T, u = simulate(f, T0)
        ax[k, 0].plot(th, u * 100, "r--"); ax[k, 0].set_ylabel("u(t) (%)")
        ax[k, 0].set_title(f"Sinyal kontrol: {nama}"); ax[k, 0].grid(True)
        ax[k, 1].plot(th, T, "b"); ax[k, 1].set_ylabel("T(t) (°C)")
        ax[k, 1].set_title(f"Respon suhu: {nama}"); ax[k, 1].grid(True)
        print(nama, "Tmin = %.2f, Takhir = %.2f" % (T.min(), T[-1]))
    for a in ax[-1]: a.set_xlabel("Waktu (jam)")
    plt.tight_layout(); plt.savefig("grafik_sinyal_kontrol.png", dpi=200); plt.close()

    # ---- 4. diagram sistem ----
    fig, a = plt.subplots(figsize=(9, 4.8)); a.set_xlim(0, 10); a.set_ylim(0, 6); a.axis("off")
    a.add_patch(FancyBboxPatch((3, 1.5), 3.4, 2.8, boxstyle="round,pad=0.05", fc="#dbe9ff", ec="black", lw=2))
    a.text(4.7, 3.5, "RUANGAN", ha="center", weight="bold")
    a.text(4.7, 2.8, "Stock: energi termal\n(suhu T, kapasitas C)", ha="center", fontsize=9)
    def arrow(p, q, txt, tp, col="black"):
        a.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=18, lw=2, color=col))
        a.text(*tp, txt, ha="center", fontsize=9, color=col)
    arrow((0.6, 3.6), (3, 3.6), "Aliran panas dinding\nUA (T_out − T)", (1.8, 4.15))
    arrow((4.7, 5.4), (4.7, 4.35), "Beban internal Q_int (gangguan)", (4.7, 5.6), "tab:orange")
    arrow((6.4, 2.4), (9.4, 2.4), "Pendinginan AC\nQ_ac = Q_max·u(t)", (7.9, 2.95), "tab:blue")
    a.text(1.8, 2.9, "T_out (konstanta)", ha="center", fontsize=9)
    a.text(7.9, 1.3, "u(t): sinyal kontrol (0–1)", ha="center", fontsize=9, color="tab:blue")
    a.text(4.7, 0.8, "Output: T(t) = state variable = controlled variable", ha="center", fontsize=9)
    a.text(4.7, 0.3, "C dT/dt = UA (T_out − T) + Q_int − Q_max·u", ha="center", fontsize=10, weight="bold")
    plt.tight_layout(); plt.savefig("diagram_sistem.png", dpi=200); plt.close()

    # ---- 5. tabel perhitungan manual (u = 0, dt = 600 s, T0 = 32) ----
    print("\nTabel manual (dt=600 s, u=0)")
    T = 32.0
    for n in range(7):
        dTdt = (UA*(T_out-T) + Q_int)/C
        Tn = T + dTdt*600
        Ta = 37 + (32-37)*np.exp(-n*600/tau)
        print(f"{n*600:5d} {T:8.4f} {UA*(T_out-T):8.2f} {dTdt:.6f} {Tn:8.4f} {Ta:8.4f}")
        T = Tn
