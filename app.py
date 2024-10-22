from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.integrate import odeint

app = Flask(__name__)

# Parámetros fijos de la simulación
k_La = 0.1  # s^-1
C_O2_star = 8.0  # Concentración de saturación de O2 en mg/L
q_O2 = 0.5  # mg/(g·h)
mu_max = 0.3  # h^-1
K_S = 0.5  # g/L
K_O2 = 0.5  # mg/L
Y_xs = 0.4  # Rendimiento biomasa/sustrato

# Guardar solo los valores finales de la simulación previa
valores_finales = {
    'O2': None,
    'biomasa': None,
    'sustrato': None,
    'volumen': None
}

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
    global valores_finales

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

    # Si hay una simulación previa, NO usamos el último valor de la simulación anterior
    y0 = [C_O2, X, S, V]

    t = np.round(np.linspace(tiempo_inicial, tiempo_final, 101), 1)

    # Simulación
    sol = odeint(modelo, y0, t, args=(F, S_in))

    # Convertir los valores de O2 disuelto de mg/L a porcentaje de saturación
    O2_saturacion = (sol[:, 0] / C_O2_star) * 100

    # Guardar los valores finales de la simulación
    valores_finales['O2'] = sol[-1, 0]
    valores_finales['biomasa'] = sol[-1, 1]
    valores_finales['sustrato'] = sol[-1, 2]
    valores_finales['volumen'] = sol[-1, 3]

    # Devolver los resultados como JSON
    resultados = {
        'tiempo': t.tolist(),
        'O2': O2_saturacion.tolist(),
        'biomasa': sol[:, 1].tolist(),
        'sustrato': sol[:, 2].tolist(),
        'volumen': sol[:, 3].tolist()
    }
    return jsonify(resultados)

# Ruta para resetear los valores finales
@app.route('/reset', methods=['POST'])
def reset_simulation():
    global valores_finales
    valores_finales = {
        'O2': None,
        'biomasa': None,
        'sustrato': None,
        'volumen': None
    }
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    app.run(debug=True)
