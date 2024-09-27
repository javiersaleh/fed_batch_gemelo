from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.integrate import odeint

app = Flask(__name__)

# Parámetros fijos de la simulación
k_La = 0.1  # s^-1
C_O2_star = 8.0  # Concentración de saturación de O2 en mg/L (valor estándar a 25°C)
q_O2 = 0.5  # mg/(g·h)
mu_max = 0.4  # h^-1
K_S = 0.1  # g/L
K_O2 = 0.1  # mg/L, constante de saturación de oxígeno
Y_xs = 0.5  # Rendimiento biomasa/sustrato

# Función del modelo
def modelo(y, t, F, S_in):
    C_O2, X, S, V = y
    
    # Ecuación de crecimiento limitada por sustrato y oxígeno
    mu = mu_max * (S / (K_S + S)) * (C_O2 / (K_O2 + C_O2))
    
    dV_dt = F
    dX_dt = mu * X - (F / V) * X
    dS_dt = - (mu / Y_xs) * X + (F / V) * (S_in - S)
    dC_O2_dt = k_La * (C_O2_star - C_O2) - q_O2 * X - (F / V) * C_O2
    
    return [dC_O2_dt, dX_dt, dS_dt, dV_dt]

# Ruta principal que carga la página inicial
@app.route('/')
def index():
    return render_template('index.html')

# Ruta que realiza la simulación y recibe datos del usuario
@app.route('/simulate', methods=['POST'])
def simulate():
    # Recibir datos del formulario HTML
    tiempo_inicial = float(request.form['tiempo_inicial'])
    tiempo_final = float(request.form['tiempo_final'])
    C_O2_saturacion = float(request.form['C_O2'])  # O2 en porcentaje de saturación
    X = float(request.form['X'])
    S = float(request.form['S'])
    V = float(request.form['V'])
    F = float(request.form['F'])  # Flujo de alimentación
    S_in = float(request.form['S_in'])  # Concentración de sustrato en la alimentación

    # Convertir el O2 de porcentaje de saturación a mg/L
    C_O2 = (C_O2_saturacion / 100) * C_O2_star

    # Condiciones iniciales
    y0 = [C_O2, X, S, V]
    t = np.linspace(tiempo_inicial, tiempo_final, 100)

    # Simulación
    sol = odeint(modelo, y0, t, args=(F, S_in))

    # Convertir los valores de O2 disuelto de mg/L a porcentaje de saturación
    O2_saturacion = (sol[:, 0] / C_O2_star) * 100

    # Devolver los resultados como JSON
    resultados = {
        'tiempo': t.tolist(),
        'O2': O2_saturacion.tolist(),  # Enviar O2 en porcentaje
        'biomasa': sol[:, 1].tolist(),
        'sustrato': sol[:, 2].tolist(),
        'volumen': sol[:, 3].tolist()
    }
    return jsonify(resultados)

if __name__ == '__main__':
    app.run(debug=True)
