from flask import Flask, render_template, request, jsonify, session
import numpy as np
from scipy.integrate import odeint

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Necesario para usar variables de sesión

# Parámetros fijos de la simulación
k_La = 0.1  # s^-1
C_O2_star = 8.0  # Concentración de saturación de O2 en mg/L (valor estándar a 25°C)
q_O2 = 0.5  # mg/(g·h)
mu_max = 0.4  # h^-1
K_S = 0.1  # g/L
Y_xs = 0.5  # Rendimiento biomasa/sustrato

# Almacenar la simulación previa
@app.route('/reset', methods=['POST'])
def reset_simulation():
    session.clear()  # Limpiar los datos guardados
    return jsonify({'status': 'simulación reseteada'})

# Función del modelo
def modelo(y, t, F, S_in):
    C_O2, X, S, V = y
    mu = mu_max * (S / (K_S + S))
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
    C_O2 = float(request.form['C_O2'])
    X = float(request.form['X'])
    S = float(request.form['S'])
    V = float(request.form['V'])
    F = float(request.form['F'])  # Flujo de alimentación
    S_in = float(request.form['S_in'])  # Concentración de sustrato en la alimentación

    # Verificar si hay una simulación anterior guardada
    if 'tiempo' in session:
        tiempo_anterior = session['tiempo'][-1]  # Último tiempo simulado
        if tiempo_inicial <= tiempo_anterior:
            tiempo_inicial = tiempo_anterior  # Empezar desde el tiempo final anterior
        condiciones_iniciales = [
            session['O2'][-1],  # Último valor de O2
            session['biomasa'][-1],  # Último valor de biomasa
            session['sustrato'][-1],  # Último valor de sustrato
            session['volumen'][-1]  # Último valor de volumen
        ]
    else:
        # No hay simulación anterior: usar valores ingresados
        condiciones_iniciales = [C_O2, X, S, V]

    t = np.linspace(tiempo_inicial, tiempo_final, 100)

    # Simulación
    sol = odeint(modelo, condiciones_iniciales, t, args=(F, S_in))

    # Guardar los resultados en la sesión
    if 'tiempo' in session:
        session['tiempo'] += t.tolist()
        session['O2'] += sol[:, 0].tolist()
        session['biomasa'] += sol[:, 1].tolist()
        session['sustrato'] += sol[:, 2].tolist()
        session['volumen'] += sol[:, 3].tolist()
    else:
        session['tiempo'] = t.tolist()
        session['O2'] = sol[:, 0].tolist()
        session['biomasa'] = sol[:, 1].tolist()
        session['sustrato'] = sol[:, 2].tolist()
        session['volumen'] = sol[:, 3].tolist()

    # Devolver los resultados como JSON
    resultados = {
        'tiempo': session['tiempo'],
        'O2': session['O2'],
        'biomasa': session['biomasa'],
        'sustrato': session['sustrato'],
        'volumen': session['volumen']
    }
    return jsonify(resultados)

if __name__ == '__main__':
    app.run(debug=True)
