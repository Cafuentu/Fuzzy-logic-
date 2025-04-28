import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# ========================
# === PARÁMETROS GLOBALES
# ========================

EP = (-12, 12)
TP = (-12, 12)
DH = (-1, 1) #rango normalizado, uso un factor de escala 12 para volver al rango real [-12,12] linea 255

PO = 700           # Presión objetivo
P_0 = 720          # Presión inicial AQUÍ VARIAR EL PARAMETRO DE ENTRADA
K = 0.8            # Ganancia de la planta
ITE = 50   # Ciclos de simulación
P_DH = 100    # Resolución del universo de discurso
n_reglas = 17    # numero de reglas
# Constante de planta (modificable, debe estar entre 0 y 1)
c = 0.9  # se puede cambiar por otro valor como 0.5 o 0.9
x_dh = np.linspace(DH[0], DH[1], P_DH)

# ========================
# === METODO
# ========================

# mean      = algoritmo promedio de los supremos
# centroide = algoritmo centro de gravedad
# altura    = algoritmo de altura

seleccionar_metodo = "centroide"  # <--- CAMBIAR AQUÍ PARA SELECCIONAR MÉTODO por centroide, altura O supremos
print(f"=== SIMULADOR CLD - MÉTODO: {seleccionar_metodo.upper()} ===")

# ========================
# === CONJUNTOS DIFUSOS
# ========================

c_difusos = {
    "Ng": (-1.0, -1.0, -0.8, -0.5),
    "Nm": (-0.8, -0.5, -0.4, -0.2),
    "Np": (-0.4, -0.3, -0.2, -0.1),
    "Ni": (-0.2, -0.1,  0.0,  0.0),
    "Ce": (-0.2,  0.0,  0.0,  0.2),
    "Pi": ( 0.0,  0.0,  0.1,  0.2),
    "Pp": ( 0.1,  0.2,  0.3,  0.4),
    "Pm": ( 0.2,  0.4,  0.5,  0.8),
    "Pq": ( 0.5,  0.8,  1.0,  1.0),
}

# ========================
# === FUNCION TRAPEZOIDAL
# ========================

def trapmf(x, a, b, c, d):
    x = np.asarray(x)
    u = np.zeros_like(x)
    
    # Caso 1: a == b (pendiente izquierda vertical)
    if np.isclose(a, b):
        idx_izq = (x >= a) & (x <= c)
        u[idx_izq] = 1.0  # Máxima pertenencia desde a hasta c
    else:
        # Pendiente izquierda normal
        idx_asc = (x >= a) & (x < b)
        u[idx_asc] = (x[idx_asc] - a) / (b - a + 1e-6)
    
    # Caso 2: c == d (pendiente derecha vertical)
    if np.isclose(c, d):
        idx_der = (x >= b) & (x <= d)
        u[idx_der] = 1.0  # Máxima pertenencia desde b hasta d
    else:
        # Pendiente derecha normal
        idx_desc = (x > c) & (x <= d)
        u[idx_desc] = (d - x[idx_desc]) / (d - c + 1e-6)
    
    # Parte plana (si existe)
    idx_plano = (x >= b) & (x <= c)
    u[idx_plano] = 1.0
    
    return np.clip(u, 0.0, 1.0)
# ========================
# === OPERADOR
# ========================

def de_a(A, B):
    a1, b1, c1, d1 = c_difusos[A]
    a2, b2, c2, d2 = c_difusos[B]
    return (min(a1, a2), min(b1, b2), max(c1, c2), max(d1, d2))

# ========================
# === ALGORITMO UTILIZADO DEFINIDO EN METODO 
# ========================
def desdifusificar(x, ux, metodo= seleccionar_metodo):
    if metodo == "mean":
        max_ux = np.max(ux)
        if max_ux == 0:
            return 0
        x_maximos = x[ux == max_ux]
        return np.mean(x_maximos)
    elif metodo == "centroide":
        return np.sum(x * ux) / (np.sum(ux) + 1e-6)
    elif metodo == "altura":
        return x[np.argmax(ux)]
    else:
        raise ValueError("Método no reconocido")

# ========================
# === REGLAS DIFUSAS
# ========================

reglas_difusas = [
    {"EP": "Ng", "TP": ["Ng", "Pp"],       "DH": "Pq"},      #R1
    {"EP": ["Ng", "Nm"], "TP": ["Ng", "Np"], "DH": "Pm"},    #R2
    {"EP": "Np", "TP": ["Np", "Pi"],       "DH": "Pm"},      #R3
    {"EP": "Ni", "TP": ["Ng", "Nm"],       "DH": "Pm"},      #R4
    {"EP": "Ni", "TP": ["Pm", "Pq"],       "DH": "Np"},      #R5
    {"EP": ["Ni", "Pi"], "TP": "Ce",       "DH": "Ce"},      #R6
    {"EP": "Pi", "TP": ["Ng", "Nm"],       "DH": "Pp"},      #R7
    {"EP": "Pi", "TP": ["Pm", "Pq"],       "DH": "Nm"},      #R8
    {"EP": "Pp", "TP": ["Np", "Pq"],       "DH": "Nm"},      #R9
    {"EP": ["Pm", "Pq"], "TP": ["Pp", "Pq"], "DH": "Nm"},    #R10
    {"EP": "Pq", "TP": ["Np", "Pq"],       "DH": "Ng"},      #R11
    {"EP": "Ni", "TP": "Pp",               "DH": "Ce"},      #R12
    {"EP": "Ni", "TP": "Np",               "DH": "Pp"},      #R13
    {"EP": "Pi", "TP": "Np",               "DH": "Ce"},      #R14
    {"EP": "Pi", "TP": "Pp",               "DH": "Np"},      #R15
    {"EP": ["Ng", "Np"], "TP": ["Pm", "Pq"], "DH": "Pq"},    #R16
    {"EP": ["Pp", "Pq"], "TP": ["Ng", "Nm"], "DH": "Ng"},    #R17
]


# ========================
# === MONITOREO DE PERTINENCIA 
# ========================

mu_ep_test = trapmf(1.0, *c_difusos["Pq"])
print("mu(1.0 in Pq):", mu_ep_test)

mu_tp_test = trapmf(1.0, *de_a("Np", "Pq"))
print("mu(1.0 in Np/Pq):", mu_tp_test)


# ========================
# === MOTOR DE INFERENCIA
# ========================

def evaluar_regla(ep_val, tp_val, regla, x_dh):
    ep_set = de_a(*regla["EP"]) if isinstance(regla["EP"], list) else c_difusos[regla["EP"]]  # Obtener el conjunto difuso de EP (Error de Presión)
    tp_set = de_a(*regla["TP"]) if isinstance(regla["TP"], list) else c_difusos[regla["TP"]]  # Obtener el conjunto difuso de TP (Tasa de Cambio de la Presión)
    u_ep = trapmf(ep_val, *ep_set)  # Evaluar el grado de pertenencia del valor actual de EP al conjunto difuso correspondiente
    u_tp = trapmf(tp_val, *tp_set)  # Evaluar el grado de pertenencia del valor actual de TP al conjunto difuso correspondiente
    activacion = min(u_ep, u_tp) # Determinar la activación de la regla como el mínimo entre u_ep y u_tp (operador AND difuso)
    u_dh = trapmf(x_dh, *c_difusos[regla["DH"]]) # Obtener la función de pertenencia del consecuente (salida ΔH)
    return np.fmin(activacion, u_dh) # Recortar la función de salida aplicando el nivel de activación (implicación tipo Mamdani)

def inferencia_cld(ep_val, tp_val, reglas, x_dh, metodo= seleccionar_metodo ):
    u_salida_total = np.zeros_like(x_dh)
    for regla in reglas:
        u_recortada = evaluar_regla(ep_val, tp_val, regla, x_dh)
        u_salida_total = np.fmax(u_salida_total, u_recortada)
    return desdifusificar(x_dh, u_salida_total, metodo), u_salida_total

# ========================
# === CREAR MAPA
# ========================

# Definiciones
etiquetas_EP = ["Ng", "Nm", "Np", "Ni", "Ce", "Pi", "Pp", "Pm", "Pq"]
etiquetas_TP = ["Ng", "Nm", "Np", "Ni", "Ce", "Pi", "Pp", "Pm", "Pq"]

mapa = np.full((len(etiquetas_EP), len(etiquetas_TP)), "", dtype=object)

for i, ep in enumerate(etiquetas_EP):
    for j, tp in enumerate(etiquetas_TP):
        salidas = []
        for regla in reglas_difusas:
            ep_match = ep in regla["EP"] if isinstance(regla["EP"], list) else ep == regla["EP"]
            tp_match = tp in regla["TP"] if isinstance(regla["TP"], list) else tp == regla["TP"]
            if ep_match and tp_match:
                salidas.append(regla["DH"])
        mapa[i, j] = "/".join(salidas) if salidas else "-"

# Visualización del mapa de reglas simbólico bonito
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xticks(np.arange(len(etiquetas_TP)))
ax.set_yticks(np.arange(len(etiquetas_EP)))
ax.set_xticklabels(etiquetas_TP, fontsize=9, rotation=45, ha="right")
ax.set_yticklabels(etiquetas_EP, fontsize=9)

# Fondo de colores por cantidad de reglas activadas
cantidad_reglas = np.vectorize(lambda x: x.count("/") + 1 if x != "-" else 0)(mapa).astype(int)
cmap = plt.cm.viridis
im = ax.imshow(cantidad_reglas, cmap=cmap)

# Asignación y Mostrar etiquetas
for i in range(len(etiquetas_EP)):
    for j in range(len(etiquetas_TP)):
        text = mapa[i, j]
        color = "white" if cantidad_reglas[i, j] >= 2 else "black"
        ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

# Títulos y leyenda
ax.set_xlabel("TP (Tasa de Cambio)", fontsize=11)
ax.set_ylabel("EP (Error de Presión)", fontsize=11)
ax.set_title("Mapa de Reglas Difusas EP x TP → DH", fontsize=13)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Cantidad de Reglas Activadas", fontsize=10)

plt.tight_layout()
plt.show()

# ========================
# === SIMULACIÓN CORREGIDA 
# ========================

P_hist, EP_hist, TP_hist, DH_hist = [P_0], [], [], []

print(f"{'Iter':<5} {'P(t)':>8} {'EP':>8} {'TP':>8} {'EP_norm':>10} {'TP_norm':>10} {'dH':>8} {'P(t+1)':>10}")
print("=" * 70)
resultados = {}

for t in range(ITE):
    P_actual = P_hist[-1]
    EP_t = P_actual - PO  # EP(t) = P(t) - PO
    print(f"error de presion = {EP_t:.2f}")
    # Cálculo correcto de TP(t)
    if t == 0:
        EP_prev = 0  # asumimos EP(t-1) = 0
        TP_t = EP_t - EP_prev
    else:
        TP_t = EP_t - EP_hist[-1]
        print(f"tasa de cambio = {TP_t:.2f}")


    # Normalización
    ep_norm = np.clip(EP_t / 12, -1, 1)
    tp_norm = np.clip(TP_t / 12, -1, 1)

    # Inicializar matriz de activación real
    if t == 0:
        activaciones_reales = []

    activaciones_t = []

    for regla in reglas_difusas:
        ep_set = de_a(*regla["EP"]) if isinstance(regla["EP"], list) else c_difusos[regla["EP"]]
        tp_set = de_a(*regla["TP"]) if isinstance(regla["TP"], list) else c_difusos[regla["TP"]]
        u_ep = trapmf(ep_norm, *ep_set)
        u_tp = trapmf(tp_norm, *tp_set)
        activaciones_t.append(min(u_ep, u_tp))

    activaciones_reales.append(activaciones_t)

    # Condición inicial para DP(t-1)
    if t == 0:
        DP_prev = 0
    else:
        DP_prev = P_hist[-1] - P_hist[-2]

    # Inferencia y evolución de la planta
    DH_t, _ = inferencia_cld(ep_norm, tp_norm, reglas_difusas, x_dh, metodo=seleccionar_metodo)
    DH_t1 = DH_t * 12 # donde DH_t normlaizado y DH_t1 real 
    DP_t = K * DH_t1 + c * DP_prev
    P_next = P_actual + DP_t

    # Registro
    EP_hist.append(EP_t)
    TP_hist.append(TP_t)
    DH_hist.append(DH_t)
    P_hist.append(P_next)

    # Output de seguimiento
    print(f"{t:<5} {P_actual:>8.2f} {EP_t:>8.2f} {TP_t:>8.2f} {ep_norm:>10.2f} {tp_norm:>10.2f} {DH_t:>8.2f} {P_next:>10.2f}")

# Guardar resultados
resultados[seleccionar_metodo] = P_hist


# ========================
# === MÉTRICAS DE ERROR
# ========================

MAE = np.mean(np.abs(np.array(P_hist[:-1]) - PO))
MSE = np.mean((np.array(P_hist[:-1]) - PO)**2)
print("\n=== Métricas de desempeño ===")
print(f"MAE (Error Absoluto Medio): {MAE:.2f}")
print(f"MSE (Error Cuadrático Medio): {MSE:.2f}")
# ========================
# === VISUALIZACIÓN
# ========================

plt.figure(figsize=(12, 6))

# Presión
plt.subplot(2, 1, 1)
plt.plot(P_hist, label="Presión (P)")
plt.axhline(PO, color='r', linestyle='--', label=f"Presión Objetivo (PO = {PO})")
plt.text(len(P_hist)-1, PO+2, f'PO = {PO}', color='r')
plt.ylabel("Presión")
plt.title(f"Presión vs Iteraciones | MAE={MAE:.2f} | MSE={MSE:.2f}")
plt.legend()
plt.grid()

# Calor aplicado
plt.subplot(2, 1, 2)
plt.plot(DH_hist, label="ΔH aplicado", color='orange')
plt.xlabel("Iteraciones")
plt.ylabel("ΔH (calor aplicado)")
plt.title("Calor aplicado vs Iteraciones")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# === Mapa de reglas EP x TP ===
ep_vals = np.linspace(-1, 1, 41)
tp_vals = np.linspace(-1, 1, 41)
mapa = np.zeros((len(ep_vals), len(tp_vals)))

for i, ep in enumerate(ep_vals):
    for j, tp in enumerate(tp_vals):
        DH_val, _ = inferencia_cld(ep, tp, reglas_difusas, x_dh, metodo= seleccionar_metodo)
        mapa[i, j] = DH_val

plt.figure(figsize=(8, 6))
plt.imshow(mapa, extent=[-1, 1, -1, 1], origin='lower', aspect='auto', cmap='coolwarm')
plt.colorbar(label='DH desfuzzificado')
plt.xlabel('TP (Tasa de Cambio del Error)')
plt.ylabel('EP (Error de Presión)')
plt.title('Mapa de Reglas EP x TP → DH')
plt.grid(True)
plt.show()

# ========================
# === MAPA DE REGLAS EP x TP
# ========================

ep_vals = np.linspace(-1, 1, 41)
tp_vals = np.linspace(-1, 1, 41)
mapa_etiquetas = np.empty((len(ep_vals), len(tp_vals)), dtype=object)

# Evaluar qué etiqueta (DH) se activa más en cada combinación de EP, TP
for i, ep in enumerate(ep_vals):
    for j, tp in enumerate(tp_vals):
        activaciones = []
        etiquetas = []

        for regla in reglas_difusas:
            ep_set = de_a(*regla["EP"]) if isinstance(regla["EP"], list) else c_difusos[regla["EP"]]
            tp_set = de_a(*regla["TP"]) if isinstance(regla["TP"], list) else c_difusos[regla["TP"]]

            u_ep = trapmf(ep, *ep_set)
            u_tp = trapmf(tp, *tp_set)
            a = min(u_ep, u_tp)

            activaciones.append(a)
            etiquetas.append(regla["DH"])

        if max(activaciones) == 0:
            mapa_etiquetas[i, j] = "-"  # sin activación
        else:
            idx_max = np.argmax(activaciones)
            mapa_etiquetas[i, j] = etiquetas[idx_max]

# ========================
# === MAPA COLORIDO DE ETIQUETAS (MEJORADO)
# ========================


etiquetas_unicas = sorted(set(label for row in mapa_etiquetas for label in row))
etiqueta_a_idx = {label: idx for idx, label in enumerate(etiquetas_unicas)}
mapa_numerico = np.vectorize(etiqueta_a_idx.get)(mapa_etiquetas)
colormap_base = plt.colormaps.get_cmap("tab20")  # sin el segundo argumento
colors = colormap_base(np.linspace(0, 1, len(etiquetas_unicas)))  # extraemos N colores
cmap = mcolors.ListedColormap(colors)
fig, ax = plt.subplots(figsize=(10, 8))

# Mostrar el mapa (Y no se invierte ahora)
im = ax.imshow(
    mapa_numerico,
    cmap=cmap,
    extent=[-1, 1, -1, 1],
    origin="lower",  # Ahora EP = -1 está abajo y EP = 1 arriba
    aspect='auto',
    interpolation='nearest'
)

# Ticks y etiquetas
ax.set_xticks(np.linspace(-1, 1, 5))
ax.set_yticks(np.linspace(-1, 1, 5))
ax.set_xlabel("TP (Tasa de cambio)")
ax.set_ylabel("EP (Error de presión)")
ax.set_title("Mapa de Reglas EP x TP (etiquetas de ΔH)")

# Superponer etiquetas DH en cada celda
for i, ep in enumerate(ep_vals):
    for j, tp in enumerate(tp_vals):
        dh = mapa_etiquetas[i, j]
        if dh != "-":
            ax.text(tp, ep, dh, ha='center', va='center', fontsize=6, color='black')

# Barra de color con etiquetas DH
cbar = plt.colorbar(im, ticks=range(len(etiquetas_unicas)))
cbar.ax.set_yticklabels(etiquetas_unicas)
cbar.set_label("Etiqueta de salida DH (DH)")

# Superponer trayectoria real
ep_tray = [np.clip(ep / 12, -1, 1) for ep in EP_hist]
tp_tray = [np.clip(tp / 12, -1, 1) for tp in TP_hist]

ax.plot(tp_tray, ep_tray, 'w.-', label='Trayectoria', linewidth=1, markersize=4)
ax.legend()

plt.grid(False)
plt.tight_layout()
plt.show()

# ========================
# === MINI TEST PARA ENTEDER LA ACTIVACIÓN DE REGLAS 
# ========================
ep_test = 1.0
tp_test = 1.0

for idx, regla in enumerate(reglas_difusas):
    ep_set = de_a(*regla["EP"]) if isinstance(regla["EP"], list) else c_difusos[regla["EP"]]
    tp_set = de_a(*regla["TP"]) if isinstance(regla["TP"], list) else c_difusos[regla["TP"]]
    u_ep = trapmf(ep_test, *ep_set)
    u_tp = trapmf(tp_test, *tp_set)
    a = min(u_ep, u_tp)
    print(f"[{idx}] EP: {regla['EP']}, TP: {regla['TP']} -> mu_ep={u_ep:.3f}, mu_tp={u_tp:.3f}, activacion={a:.3f}")


# ========================
# === MOSTRAR LAS REGLAS ACTIVAS EN LA ITERACIÓN 
# ========================

activaciones_reales = np.array(activaciones_reales)
plt.figure(figsize=(12, 6))
plt.imshow(activaciones_reales.T, aspect='auto', cmap='YlGnBu', interpolation='nearest')
plt.colorbar(label="Nivel de activación")
plt.xlabel("Iteración")
plt.ylabel("Regla")
plt.title("Secuencia real de Disparo de Reglas por Iteración")
plt.yticks(ticks=np.arange(n_reglas), labels=[f"R{i+1}" for i in range(n_reglas)])
plt.grid(False)
plt.tight_layout()
plt.show()
