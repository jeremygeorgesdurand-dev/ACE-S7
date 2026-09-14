"""
Projet ACE S7 - Asservissement de temperature d'une resistance chauffante via une thermistance
Partie 2 : Modele thermique
    2.1 Bilan thermique / equation de la chaleur du systeme
    2.2 Simulation numerique du modele dynamique (Python)
    2.3 Analyse de sensibilite aux parametres thermiques

Modele utilise : systeme a parametres localises (modele 0D / "lumped capacitance"),
valable si le nombre de Biot Bi = h*Lc/lambda << 1 (bonne conductivite de l'aluminium,
piece de petite epaisseur -> hypothese raisonnable ici, a verifier a posteriori avec
les mesures reelles).

Bilan d'energie sur l'ensemble {resistance chauffante DBK HP03-1/04-24 + profile
aluminium support + thermistance EPCOS B57045K473K collee en surface} :

    C_th * dT/dt = P_elec(t) - (T(t) - T_amb) / R_th

avec :
    C_th = m * cp        [J/K]   capacite thermique de l'ensemble
    R_th = 1 / (h * A)   [K/W]   resistance thermique vers l'ambiant (convection naturelle)
    P_elec(t)            [W]     puissance electrique dissipee dans la resistance chauffante
    T_amb                [degC]  temperature ambiante

Reponse a un echelon de puissance P0 (a t=0, T(0)=T_amb) :

    T(t) = T_amb + P0 * R_th * (1 - exp(-t / tau)),   tau = R_th * C_th

Regime etabli : T_ss = T_amb + P0 * R_th
Temps de reponse a 63% : tau ; a 95% : 3*tau ; a 99% : 4.6*tau

IMPORTANT : les valeurs numeriques ci-dessous (masse, surface d'echange, coefficient
de convection h) sont des ESTIMATIONS de depart, faute de mesures. Elles sont a
recaler avec les valeurs reelles (pesee de l'ensemble, mesure des dimensions) et,
surtout, avec l'identification experimentale (reponse a un echelon de puissance
mesuree sur le banc) demandee en 2.3 du cahier des charges.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Dossier de sortie des figures : toujours a cote de ce script, quel que soit le
# repertoire courant (cwd) depuis lequel le script est lance (ex: lance depuis "/",
# qui est en lecture seule sur macOS -> sinon erreur "Read-only file system").
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# 1. Parametres physiques (valeurs par defaut - A AJUSTER avec le systeme reel)
# ----------------------------------------------------------------------------

# Resistance chauffante DBK HP03-1/04-24 (datasheet constructeur)
P_MAX = 15.0          # W, puissance maximale de l'element chauffant PTC

# Ensemble {resistance + profile alu support + thermistance}
M_ASSEMBLY = 0.05      # kg, masse estimee de l'ensemble (a peser reellement)
CP_ALU = 897.0          # J/(kg.K), capacite thermique massique de l'aluminium
C_TH = M_ASSEMBLY * CP_ALU   # J/K

# Echange thermique avec l'air ambiant (convection naturelle)
A_EXCHANGE = 0.010     # m^2, surface d'echange estimee (~100 cm^2, a mesurer sur le profile reel)
H_CONV = 8.0            # W/(m^2.K), coefficient de convection naturelle dans l'air (ordre de grandeur 5-15)
R_TH = 1.0 / (H_CONV * A_EXCHANGE)   # K/W

T_AMB = 20.0            # degC, temperature ambiante de reference

TAU = R_TH * C_TH        # s, constante de temps thermique


def print_summary():
    print("=" * 70)
    print("Parametres thermiques (valeurs de depart)")
    print("=" * 70)
    print(f"  C_th (capacite thermique)      = {C_TH:8.2f} J/K")
    print(f"  R_th (resistance thermique)     = {R_TH:8.3f} K/W")
    print(f"  tau = R_th * C_th               = {TAU:8.1f} s  ({TAU/60:.1f} min)")
    print(f"  T_ss a Pmax ({P_MAX} W)          = {T_AMB + R_TH * P_MAX:8.1f} degC")
    print("=" * 70)


# ----------------------------------------------------------------------------
# 2.1 / 2.2 : modele dynamique et resolution numerique
# ----------------------------------------------------------------------------

def thermal_ode(t, T, P_func, R_th, C_th, T_amb):
    """dT/dt = (P(t) - (T-T_amb)/R_th) / C_th"""
    P = P_func(t)
    return [(P - (T[0] - T_amb) / R_th) / C_th]


def analytical_step_response(t, P0, R_th, C_th, T_amb, T0=None):
    """Solution analytique pour un echelon de puissance P0 applique a t=0."""
    if T0 is None:
        T0 = T_amb
    tau = R_th * C_th
    T_ss = T_amb + R_th * P0
    return T_ss + (T0 - T_ss) * np.exp(-t / tau)


def simulate_step(P0=P_MAX, t_end=None, R_th=R_TH, C_th=C_TH, T_amb=T_AMB, n_points=2000):
    """Simule la reponse a un echelon de puissance P0 (identification par essai indiciel)."""
    tau = R_th * C_th
    if t_end is None:
        t_end = 6 * tau  # regime etabli a ~99.7%
    t_eval = np.linspace(0, t_end, n_points)

    P_func = lambda t: P0
    sol = solve_ivp(
        thermal_ode, [0, t_end], [T_amb], t_eval=t_eval,
        args=(P_func, R_th, C_th, T_amb), method="RK45", rtol=1e-8, atol=1e-8,
    )
    return sol.t, sol.y[0]


def simulate_euler_explicit(P0=P_MAX, t_end=None, dt=1.0, R_th=R_TH, C_th=C_TH, T_amb=T_AMB):
    """
    Integration par Euler explicite -- correspond au schema qui sera utilise dans le
    firmware STM32 (tache periodique, pas dt fixe) : T[k+1] = T[k] + dt/C_th * (P - (T[k]-Tamb)/R_th)
    Utile pour verifier la stabilite/precision du pas choisi pour la commande numerique
    (condition de stabilite : dt << tau).
    """
    tau = R_th * C_th
    if t_end is None:
        t_end = 6 * tau
    n = int(t_end / dt) + 1
    t = np.arange(n) * dt
    T = np.zeros(n)
    T[0] = T_amb
    for k in range(n - 1):
        T[k + 1] = T[k] + dt / C_th * (P0 - (T[k] - T_amb) / R_th)
    return t, T


def simulate_creneaux(P0, periods_s, t_end, R_th=R_TH, C_th=C_TH, T_amb=T_AMB, n_points=4000):
    """
    Reponse a une consigne en creneaux (ON/OFF alternes) -- profil typique utilise pour
    l'identification experimentale des parametres thermiques (montees/descentes successives).
    periods_s : duree (s) de chaque palier ON puis OFF.
    """
    def P_func(t):
        phase = t % (2 * periods_s)
        return P0 if phase < periods_s else 0.0

    t_eval = np.linspace(0, t_end, n_points)
    sol = solve_ivp(
        thermal_ode, [0, t_end], [T_amb], t_eval=t_eval,
        args=(P_func, R_th, C_th, T_amb), method="RK45", rtol=1e-8, atol=1e-8,
    )
    P_trace = np.array([P_func(tt) for tt in sol.t])
    return sol.t, sol.y[0], P_trace


# ----------------------------------------------------------------------------
# 2.3 : Analyse de sensibilite aux parametres thermiques
# ----------------------------------------------------------------------------

def sensitivity_table(variation=0.30):
    """
    Fait varier chaque parametre physique de +/- `variation` (30% par defaut) autour
    de sa valeur nominale, et mesure l'impact sur tau et T_ss (a P0=P_MAX).
    Renvoie une liste de dicts prets a etre affiches / traces.
    """
    params_nominal = dict(m=M_ASSEMBLY, cp=CP_ALU, h=H_CONV, A=A_EXCHANGE, P=P_MAX)
    results = []

    def compute(m, cp, h, A, P):
        C = m * cp
        R = 1.0 / (h * A)
        tau = R * C
        Tss = T_AMB + R * P
        return tau, Tss

    tau_nom, Tss_nom = compute(**params_nominal)

    for name in ["m", "cp", "h", "A", "P"]:
        for sign, label in [(-1, "-{:.0f}%".format(variation * 100)),
                             (+1, "+{:.0f}%".format(variation * 100))]:
            p = dict(params_nominal)
            p[name] = params_nominal[name] * (1 + sign * variation)
            tau, Tss = compute(**p)
            results.append({
                "parametre": name,
                "variation": label,
                "tau_s": tau,
                "delta_tau_pct": 100 * (tau - tau_nom) / tau_nom,
                "T_ss_degC": Tss,
                "delta_Tss_pct": 100 * (Tss - Tss_nom) / Tss_nom,
            })
    return tau_nom, Tss_nom, results


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def plot_step_response(outfile=None):
    if outfile is None:
        outfile = os.path.join(OUTPUT_DIR, "fig1_reponse_echelon.png")
    t_num, T_num = simulate_step()
    T_ana = analytical_step_response(t_num, P_MAX, R_TH, C_TH, T_AMB)
    t_euler, T_euler = simulate_euler_explicit(dt=max(1.0, TAU / 200))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t_num / 60, T_ana, "-", lw=2.5, color="#1f77b4", label="Solution analytique")
    ax.plot(t_num / 60, T_num, "--", lw=1.5, color="#ff7f0e", label="Numerique (RK45)")
    ax.plot(t_euler / 60, T_euler, ":", lw=1.5, color="#2ca02c", label="Euler explicite (pas dt=tau/200)")
    ax.axhline(T_AMB + R_TH * P_MAX, color="gray", ls="--", lw=0.8)
    ax.axvline(TAU / 60, color="gray", ls="--", lw=0.8)
    ax.annotate(f"tau = {TAU:.0f} s", xy=(TAU / 60, T_AMB), xytext=(TAU / 60 + 0.3, T_AMB + 5),
                fontsize=9, color="gray")
    ax.set_xlabel("Temps (min)")
    ax.set_ylabel("Temperature (degC)")
    ax.set_title(f"Reponse indicielle a un echelon de puissance P0 = {P_MAX:.0f} W")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_creneaux(outfile=None):
    if outfile is None:
        outfile = os.path.join(OUTPUT_DIR, "fig2_creneaux_identification.png")
    t, T, P = simulate_creneaux(P0=P_MAX, periods_s=TAU, t_end=6 * TAU)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 2]})
    ax1.step(t / 60, P, where="post", color="#d62728")
    ax1.set_ylabel("P (W)")
    ax1.set_title("Essai en creneaux (identification experimentale des parametres thermiques)")
    ax1.grid(alpha=0.3)

    ax2.plot(t / 60, T, color="#1f77b4", lw=1.8)
    ax2.set_xlabel("Temps (min)")
    ax2.set_ylabel("Temperature (degC)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_sensitivity(outfile=None, variation=0.30):
    if outfile is None:
        outfile = os.path.join(OUTPUT_DIR, "fig3_sensibilite.png")
    tau_nom, Tss_nom, results = sensitivity_table(variation)

    labels = ["m (masse)", "cp (chaleur massique)", "h (convection)", "A (surface)", "P (puissance)"]
    keys = ["m", "cp", "h", "A", "P"]

    tau_low = [next(r["delta_tau_pct"] for r in results if r["parametre"] == k and r["variation"].startswith("-")) for k in keys]
    tau_high = [next(r["delta_tau_pct"] for r in results if r["parametre"] == k and r["variation"].startswith("+")) for k in keys]
    tss_low = [next(r["delta_Tss_pct"] for r in results if r["parametre"] == k and r["variation"].startswith("-")) for k in keys]
    tss_high = [next(r["delta_Tss_pct"] for r in results if r["parametre"] == k and r["variation"].startswith("+")) for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    y = np.arange(len(labels))

    for ax, low, high, title in [
        (ax1, tau_low, tau_high, "Sensibilite de tau (constante de temps)"),
        (ax2, tss_low, tss_high, "Sensibilite de T_ss (regime etabli)"),
    ]:
        ax.barh(y, high, color="#1f77b4", label=f"+{variation*100:.0f}%", alpha=0.85)
        ax.barh(y, low, color="#ff7f0e", label=f"-{variation*100:.0f}%", alpha=0.85)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Variation (%)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile, tau_nom, Tss_nom, results


if __name__ == "__main__":
    print_summary()
    f1 = plot_step_response()
    f2 = plot_creneaux()
    f3, tau_nom, Tss_nom, results = plot_sensitivity()

    print("\nAnalyse de sensibilite (variation +/-30% autour du nominal) :")
    print(f"  Nominal : tau = {tau_nom:.1f} s, T_ss = {Tss_nom:.1f} degC")
    for r in results:
        print(f"  {r['parametre']:>3s} {r['variation']:>5s} -> "
              f"tau={r['tau_s']:7.1f}s ({r['delta_tau_pct']:+6.1f}%)   "
              f"T_ss={r['T_ss_degC']:6.1f}degC ({r['delta_Tss_pct']:+6.1f}%)")

    print(f"\nFigures generees : {f1}, {f2}, {f3}")
