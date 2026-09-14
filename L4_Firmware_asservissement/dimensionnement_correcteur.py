"""
Projet ACE - Asservissement de temperature d'une resistance chauffante via thermistance
Partie 3 : Dimensionnement du correcteur

Choix du correcteur : PI numerique (voir justification dans le message / le rapport).
Methode de reglage : compensation du pole dominant (Ti = tau), tres classique pour un
procede du premier ordre sans retard pur -- cf cours d'automatique (methode dite "IMC"
ou "pole cancellation").

Modele de la chaine directe (duty cycle delta -> temperature T) :

    G(s) = K / (1 + tau*s)          avec   K   = Pmax * Rth   [degC, pour delta=1]
                                             tau = Rth * C      [s]
                                             Pmax = Umax^2 / R_elec   (puissance max, delta=1)

    delta in [0, 1] est le rapport cyclique de la PWM qui pilote le MOSFET de puissance.
    Puisque la resistance est purement resistive et pilotee en PWM basse frequence
    (interrupteur ON/OFF), la puissance moyenne dissipee est LINEAIRE en delta :
        P_moy = delta * Pmax
    -> le systeme {delta -> T} est bien un premier ordre lineaire, comme {P -> T}.

Correcteur PI continu : C(s) = Kp * (1 + 1/(Ti*s))
En choisissant Ti = tau (le zero du correcteur compense le pole du procede) :

    C(s)*G(s) = Kp*K / (tau*s)   ->   boucle fermee du 1er ordre, sans depassement :
    T_bf(s) = 1 / (1 + (tau/(Kp*K))*s)   =>   tau_bf = tau / (Kp*K)

Donc pour choisir une constante de temps en boucle fermee tau_bf (plus rapide que tau,
typiquement tau_bf = tau/3 a tau/10 selon la marge dispo sur l'actionneur) :

    Kp = tau / (K * tau_bf)
    Ti = tau
    Ki = Kp / Ti
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1) Parametres identifies (a remplacer par VOS valeurs identifiees, cf modele_thermique.py)
# ---------------------------------------------------------------
T_amb = 20.0
R_elec = 4.7          # Ohm
U_max = 12.0           # V, tension d'alimentation de la resistance
Rth = 6.0               # K/W  (identifie experimentalement)
C_th = 45.0             # J/K  (identifie experimentalement)
tau = Rth * C_th         # s
Pmax = U_max ** 2 / R_elec
K = Pmax * Rth           # degC, gain statique pour delta = 1

print(f"tau = {tau:.1f} s | Pmax = {Pmax:.2f} W | K (gain statique) = {K:.1f} degC")

# ---------------------------------------------------------------
# 2) Reglage du correcteur PI (compensation de pole)
# ---------------------------------------------------------------
facteur_acceleration = 5.0             # boucle fermee 5x plus rapide que boucle ouverte
tau_bf = tau / facteur_acceleration
Ti = tau
Kp = tau / (K * tau_bf)
Ki = Kp / Ti

print(f"Reglage PI : Kp = {Kp:.5f} (duty/degC) | Ti = {Ti:.1f} s | Ki = {Ki:.6e} (duty/(degC.s))")
print(f"tau_bf visee = {tau_bf:.1f} s (soit ~{4*tau_bf/60:.1f} min pour atteindre ~98%)")


# ---------------------------------------------------------------
# 3) Simulation boucle fermee (discret, Ts << tau_bf) avec saturation + anti-windup
#    Comparaison P / PI / PID pour justifier le choix du PI
# ---------------------------------------------------------------
def simuler_boucle_fermee(Kp, Ti, Td, T_consigne=80.0, t_final=2400.0, Ts=1.0,
                           bruit_mesure_std=0.0, fan_on_at=None, Rth_forced=3.0):
    n = int(t_final / Ts)
    t = np.arange(n) * Ts
    T = np.zeros(n)
    T[0] = T_amb
    delta = 0.0
    integ = 0.0
    e_prev = 0.0
    deltas = np.zeros(n)
    rng = np.random.default_rng(1)

    for k in range(n - 1):
        Rth_k = Rth if (fan_on_at is None or t[k] < fan_on_at) else Rth_forced
        T_mes = T[k] + (rng.normal(0, bruit_mesure_std) if bruit_mesure_std else 0.0)
        e = T_consigne - T_mes

        # terme proportionnel
        P_term = Kp * e
        # terme integral (anti-windup par integration conditionnelle)
        integ_essai = integ + (Kp / Ti) * e * Ts if Ti > 0 else 0.0
        # terme derive (filtre simple sur la mesure, pas sur la consigne -> pas de "kick")
        D_term = Kp * Td * (e - e_prev) / Ts if Td > 0 else 0.0

        delta_essai = P_term + integ_essai + D_term
        delta = min(1.0, max(0.0, delta_essai))
        # anti-windup : on ne met a jour l'integrateur que si on ne sature pas,
        # ou si la saturation va dans le sens qui reduit l'erreur
        if delta == delta_essai or (delta_essai > 1.0 and e < 0) or (delta_essai < 0.0 and e > 0):
            integ = integ_essai

        deltas[k] = delta
        e_prev = e

        P_moy = delta * Pmax
        dTdt = (P_moy - (T[k] - T_amb) / Rth_k) / C_th
        T[k + 1] = T[k] + dTdt * Ts

    deltas[-1] = deltas[-2]
    return t, T, deltas


T_consigne = 80.0

# a) Comparaison P seul / PI / PID (avec un peu de bruit de mesure realiste)
bruit = 0.15  # degC, bruit de mesure typique apres conditionnement + ADC
t, T_P, d_P = simuler_boucle_fermee(Kp=Kp, Ti=1e9, Td=0.0, T_consigne=T_consigne, bruit_mesure_std=bruit)
t, T_PI, d_PI = simuler_boucle_fermee(Kp=Kp, Ti=Ti, Td=0.0, T_consigne=T_consigne, bruit_mesure_std=bruit)
t, T_PID, d_PID = simuler_boucle_fermee(Kp=Kp, Ti=Ti, Td=Ti / 10, T_consigne=T_consigne, bruit_mesure_std=bruit)

fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
axes[0].axhline(T_consigne, color="k", ls=":", label="Consigne")
axes[0].plot(t / 60, T_P, label="P seul (erreur statique)")
axes[0].plot(t / 60, T_PI, label="PI (retenu)")
axes[0].plot(t / 60, T_PID, label="PID (D bruite)")
axes[0].set_ylabel("Temperature (degC)")
axes[0].set_title(f"Reponse en boucle fermee - consigne {T_consigne} degC")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

axes[1].plot(t / 60, d_P, label="P seul")
axes[1].plot(t / 60, d_PI, label="PI (retenu)")
axes[1].plot(t / 60, d_PID, label="PID (D bruite)")
axes[1].set_xlabel("temps (min)"); axes[1].set_ylabel("Rapport cyclique (0-1)")
axes[1].set_title("Commande (duty cycle PWM) - le PID est bruite par le terme derive")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("comparaison_P_PI_PID.png", dpi=150)
print("Figure sauvegardee : comparaison_P_PI_PID.png")

# b) Reponse PI retenue + perturbation ventilateur (convection forcee) en cours d'essai
t2, T2, d2 = simuler_boucle_fermee(Kp=Kp, Ti=Ti, Td=0.0, T_consigne=T_consigne,
                                    bruit_mesure_std=bruit, fan_on_at=1200.0, Rth_forced=3.0)
fig2, ax2 = plt.subplots(figsize=(8, 4.5))
ax2.axhline(T_consigne, color="k", ls=":", label="Consigne")
ax2.plot(t2 / 60, T2, label="T (PI)")
ax2.axvline(1200 / 60, color="r", ls="--", alpha=0.6, label="Ventilateur ON")
ax2.set_xlabel("temps (min)"); ax2.set_ylabel("Temperature (degC)")
ax2.set_title("Robustesse du PI face a une perturbation (convection forcee)")
ax2.legend(); ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig("robustesse_PI_ventilateur.png", dpi=150)
print("Figure sauvegardee : robustesse_PI_ventilateur.png")

# Indicateurs de performance simples
def temps_montee_98(t, T, T_cons, T0=T_amb):
    cible = T0 + 0.98 * (T_cons - T0)
    idx = np.argmax(T >= cible)
    return t[idx] if T[idx] >= cible else np.nan

print(f"Temps pour atteindre 98% de la consigne (PI) : {temps_montee_98(t, T_PI, T_consigne)/60:.1f} min "
      f"(vise ~{4*tau_bf/60:.1f} min)")
erreur_statique_P = T_consigne - T_P[-1]
print(f"Erreur statique en P seul (justifie le besoin d'un terme integral) : {erreur_statique_P:.1f} degC")
