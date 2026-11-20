import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configuration de la page web
st.set_page_config(
    page_title="Simulateur Cinématique & Statique - Pelle",
    layout="wide"
)

# ==========================================
# 1. FONCTIONS MATHÉMATIQUES & CINÉMATIQUES
# ==========================================
def moment_2d(r_vec, F_vec):
    rx, ry = r_vec
    fx, fy = F_vec
    return rx * fy - ry * fx

def angle_vec(v):
    vx, vy = v
    return np.arctan2(vy, vx)

def tracer_segment(ax, P1, P2, **kwargs):
    x1, y1 = P1
    x2, y2 = P2
    ax.plot([x1, x2], [y1, y2], **kwargs)

def etat_verin(F):
    if F > 1e-6:
        return "Poussée"
    elif F < -1e-6:
        return "Traction"
    return "Nul"

def intersection_cercles(C1, r1, C2, r2, P_ref):
    d = np.linalg.norm(C2 - C1)
    if d > (r1 + r2) or d < abs(r1 - r2) or d == 0:
        return P_ref, False
    
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(max(0.0, r1**2 - a**2))
    P0 = C1 + a * (C2 - C1) / d
    dx, dy = C2 - C1
    perp = np.array([-dy, dx]) / d
    
    sol1 = P0 + h * perp
    sol2 = P0 - h * perp
    
    if np.linalg.norm(sol1 - P_ref) < np.linalg.norm(sol2 - P_ref):
        return sol1, True
    return sol2, True

def transformer_point(P_old, pivot_old, pivot_new, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    return pivot_new + R @ (P_old - pivot_old)

def transformer_polygone(poly_old, pivot_old, pivot_new, theta):
    return np.array([transformer_point(pt, pivot_old, pivot_new, theta) for pt in poly_old])

# ==========================================
# 2. GÉOMÉTRIE DE RÉFÉRENCE
# ==========================================
O0 = np.array([0.30, 0.08])
A0 = np.array([0.43, 0.08])
A1_ref = np.array([0.30, 0.36])
B1_ref = np.array([0.25, 0.54])
O1_ref = np.array([0.54, 0.78])
B2_ref = np.array([0.54, 0.94])
C1_ref = np.array([0.71, 0.79])
O2_ref = np.array([0.78, 0.45])
D1_ref = np.array([0.85, 0.47])

poly_fleche_ref = np.array([
    [0.27, 0.08], [0.32, 0.08], A1_ref, [0.33, 0.45], O1_ref, B1_ref, [0.22, 0.44]
])
poly_balancier_ref = np.array([
    B2_ref, C1_ref, [0.80, 0.45], O2_ref, O1_ref
])
poly_godet_ref = np.array([
    O2_ref, D1_ref, [0.90, 0.41], [0.91, 0.35], [0.87, 0.32], [0.74, 0.24], [0.79, 0.35]
])

G1_ref = np.mean(poly_fleche_ref, axis=0)
G2_ref = np.mean(poly_balancier_ref, axis=0)
G3_ref = np.mean(poly_godet_ref, axis=0)

L1_init = np.linalg.norm(A1_ref - A0)
L2_init = np.linalg.norm(B2_ref - B1_ref)
L3_init = np.linalg.norm(D1_ref - C1_ref)

# ==========================================
# 3. GESTION DE L'ÉTAT & INTERFACE WEB
# ==========================================
st.title("🚜 Simulateur Cinématique & Statique - Pelle avec Poids Propre")

if "c1" not in st.session_state:
    st.session_state.c1 = 0.85
    st.session_state.c2 = 0.80
    st.session_state.c3 = 1.00

def reset_sliders():
    st.session_state.c1 = 1.00
    st.session_state.c2 = 1.00
    st.session_state.c3 = 1.00

with st.sidebar:
    st.header("🎛️ Pilotage Vérins, Masses & Efforts")
    
    st.button("🔄 Réinitialiser la posture (1.00)", on_click=reset_sliders, width="stretch")
    
    st.subheader("Course des vérins")
    c1 = st.slider("Vérin 1 - Flèche (Coef L1)", 0.60, 1.30, key="c1", step=0.01)
    c2 = st.slider("Vérin 2 - Balancier (Coef L2)", 0.60, 1.25, key="c2", step=0.01)
    c3 = st.slider("Vérin 3 - Godet (Coef L3)", 0.65, 1.35, key="c3", step=0.01)
    
    st.subheader("Masses propres (kg)")
    m1 = st.number_input("Masse Flèche m1 (kg)", value=800.0, step=50.0)
    m2 = st.number_input("Masse Balancier m2 (kg)", value=450.0, step=50.0)
    m3 = st.number_input("Masse Godet m3 (kg)", value=250.0, step=25.0)
    
    st.subheader("Effort extérieur en G3 (N)")
    Fx = st.number_input("Effort ext. Fx (N)", value=0.0, step=500.0)
    Fy = st.number_input("Effort ext. Fy (N)", value=-5000.0, step=500.0)

# ==========================================
# 4. CALCULS CINÉMATIQUES & STATIQUES
# ==========================================
L1 = L1_init * c1
L2 = L2_init * c2
L3 = L3_init * c3

g = 9.81
P1_vec = np.array([0.0, -m1 * g])
P2_vec = np.array([0.0, -m2 * g])
P3_vec = np.array([0.0, -m3 * g])
F_ext  = np.array([Fx, Fy])

# Cinématique
A1, ok1 = intersection_cercles(O0, np.linalg.norm(A1_ref - O0), A0, L1, A1_ref)
if not ok1:
    st.error("⚠️ **[ALERTE BUTÉE MÉCANIQUE]** Vérin 1 (Flèche) hors butée ! Ramenez le curseur dans la plage valide.")
    st.stop()

theta1 = angle_vec(A1 - O0) - angle_vec(A1_ref - O0)
B1 = transformer_point(B1_ref, O0, O0, theta1)
O1 = transformer_point(O1_ref, O0, O0, theta1)
G1 = transformer_point(G1_ref, O0, O0, theta1)
poly_fleche = transformer_polygone(poly_fleche_ref, O0, O0, theta1)

B2_guess = transformer_point(B2_ref, O0, O0, theta1)
B2, ok2 = intersection_cercles(O1, np.linalg.norm(B2_ref - O1_ref), B1, L2, B2_guess)
if not ok2:
    st.error("⚠️ **[ALERTE BUTÉE MÉCANIQUE]** Vérin 2 (Balancier) hors butée ! Ramenez le curseur dans la plage valide.")
    st.stop()

theta2 = angle_vec(B2 - O1) - angle_vec(B2_ref - O1_ref)
C1 = transformer_point(C1_ref, O1_ref, O1, theta2)
O2 = transformer_point(O2_ref, O1_ref, O1, theta2)
G2 = transformer_point(G2_ref, O1_ref, O1, theta2)
poly_balancier = transformer_polygone(poly_balancier_ref, O1_ref, O1, theta2)

D1_guess = transformer_point(D1_ref, O1_ref, O1, theta2)
D1, ok3 = intersection_cercles(O2, np.linalg.norm(D1_ref - O2_ref), C1, L3, D1_guess)
if not ok3:
    st.error("⚠️ **[ALERTE BUTÉE MÉCANIQUE]** Vérin 3 (Godet) hors butée ! Ramenez le curseur dans la plage valide.")
    st.stop()

theta3 = angle_vec(D1 - O2) - angle_vec(D1_ref - O2_ref)
poly_godet = transformer_polygone(poly_godet_ref, O2_ref, O2, theta3)
G3 = transformer_point(G3_ref, O2_ref, O2, theta3)

# Statique
u1 = (A1 - A0) / np.linalg.norm(A1 - A0)
u2 = (B2 - B1) / np.linalg.norm(B2 - B1)
u3 = (D1 - C1) / np.linalg.norm(D1 - C1)

M_O2 = moment_2d(G3 - O2, F_ext + P3_vec)
F3 = -M_O2 / moment_2d(D1 - O2, u3)

M_O1 = moment_2d(G3 - O1, F_ext + P3_vec) + moment_2d(G2 - O1, P2_vec)
F2 = -M_O1 / moment_2d(B2 - O1, u2)

M_O0 = moment_2d(G3 - O0, F_ext + P3_vec) + moment_2d(G2 - O0, P2_vec) + moment_2d(G1 - O0, P1_vec)
F1 = -M_O0 / moment_2d(A1 - O0, u1)

# ==========================================
# 5. AFFICHAGE DES RÉSULTATS & GRAPHIQUE
# ==========================================
col_graph, col_res = st.columns([1.6, 1])

with col_graph:
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(-0.15, 1.30)
    ax.set_ylim(-0.05, 1.25)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Efforts totaux (avec poids) : F1 = {F1:.0f} N | F2 = {F2:.0f} N | F3 = {F3:.0f} N', fontsize=10, fontweight='bold')

    # Bâti fixe & pièces rigides rouges
    ax.plot([0.09, 0.84], [0.08, 0.08], 'r-', linewidth=5)
    ax.add_patch(patches.Polygon(poly_fleche, closed=True, facecolor='red', edgecolor='red'))
    ax.add_patch(patches.Polygon(poly_balancier, closed=True, facecolor='red', edgecolor='red'))

    # Godet bleu
    ax.add_patch(patches.Polygon(poly_godet, closed=True, facecolor=(0.25, 0.45, 0.78), edgecolor='black', linewidth=0.8))

    # Vérins verts
    verin_color = (0.0, 0.55, 0.25)
    tracer_segment(ax, A0, A1, color=verin_color, linewidth=2.5)
    tracer_segment(ax, B1, B2, color=verin_color, linewidth=2.5)
    tracer_segment(ax, C1, D1, color=verin_color, linewidth=2.5)

    # Articulations (9 disques jaunes)
    articulations = np.array([O0, A0, A1, B1, O1, B2, C1, O2, D1])
    art_x, art_y = articulations.T
    ax.plot(art_x, art_y, 'o', markersize=5,
            markerfacecolor=(1, 0.95, 0), markeredgecolor=(0.85, 0.8, 0), zorder=5)

    # Centres de gravité & vecteurs poids
    centres_G = [G1, G2, G3]
    poids_vecs = [P1_vec, P2_vec, P3_vec]
    noms_G = ['$G_1$', '$G_2$', '$G_3$']
    for G_pt, P_vec, nom in zip(centres_G, poids_vecs, noms_G):
        gx, gy = G_pt
        _, py = P_vec
        ax.plot(gx, gy, 'ko', markersize=5, markerfacecolor='w', markeredgewidth=1.1, zorder=6)
        ax.text(gx + 0.015, gy + 0.015, nom, fontweight='bold', fontsize=9, zorder=6)
        if abs(py) > 0:
            ax.quiver(gx, gy, 0, -0.09, angles='xy', scale_units='xy', scale=1, color='k', width=0.005, zorder=6)

    # Vecteur effort extérieur F_ext en G3
    norm_F = np.linalg.norm(F_ext)
    if norm_F > 0:
        echelle = 0.18 / norm_F
        g3x, g3y = G3
        fx_ext, fy_ext = F_ext
        ax.quiver(g3x, g3y, fx_ext * echelle, fy_ext * echelle, angles='xy', scale_units='xy', scale=1,
                  color=(0.85, 0.1, 0.6), width=0.007, zorder=6)

    # Étiquettes des efforts sur les vérins
    m1x, m1y = 0.5 * (A0 + A1)
    m2x, m2y = 0.5 * (B1 + B2)
    m3x, m3y = 0.5 * (C1 + D1)
    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec=verin_color, lw=0.8)
    ax.text(m1x + 0.03, m1y, f'$F_1 = {F1:+.0f}$ N', color=verin_color, fontsize=8.5, fontweight='bold', bbox=bbox_props, zorder=7)
    ax.text(m2x - 0.25, m2y + 0.02, f'$F_2 = {F2:+.0f}$ N', color=verin_color, fontsize=8.5, fontweight='bold', bbox=bbox_props, zorder=7)
    ax.text(m3x + 0.03, m3y, f'$F_3 = {F3:+.0f}$ N', color=verin_color, fontsize=8.5, fontweight='bold', bbox=bbox_props, zorder=7)

    st.pyplot(fig, width="content")
    plt.close(fig)

with col_res:
    st.subheader("📊 Résultats Statiques")
    _, p1y = P1_vec
    _, p2y = P2_vec
    _, p3y = P3_vec
    st.info(f"**Poids propres :**\n- P1 = {abs(p1y):.0f} N\n- P2 = {abs(p2y):.0f} N\n- P3 = {abs(p3y):.0f} N")
    
    st.metric(label=f"Vérin 1 (L1 = {L1:.3f} m)", value=f"{F1:+.1f} N", delta=etat_verin(F1), delta_color="off")
    st.metric(label=f"Vérin 2 (L2 = {L2:.3f} m)", value=f"{F2:+.1f} N", delta=etat_verin(F2), delta_color="off")
st.metric(label=f"Vérin 3 (L3 = {L3:.3f} m)", value=f"{F3:+.1f} N", delta=etat_verin(F3), delta_color="off")