from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.integrate import odeint

app = Flask(__name__)

# Parámetros fijos de la simulación
k_La = 0.1  # s^-1
C_O2_star = 8.0  # Concentración de saturación de O2 en mg/L (valor estándar a 25°C)
q_O2 = 0.5  # mg/(g·h)
mu_max = 0.4  # h^-1
K_S = 1  # g/L
K_O2 = 1  # mg/L, constante de saturación de oxígeno
Y_xs = 0.5  # Rendimiento biomasa/sustrato

# Guardar los resultados previos
datos_guardados = {
    'tiempo': [],
    'O2': [],
    'biomasa': [],
    'sustrato': [],
    'volumen': []
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
    global datos_guardados

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

    # Si ya hay datos previos y el tiempo inicial es mayor que 0, mantener datos hasta ese punto
    if datos_guardados['tiempo']:
        # Guardamos los datos previos hasta el tiempo inicial
        indice_quiebre = np.argmax(np.array(datos_guardados['tiempo']) >= tiempo_inicial)
        
        # Usamos los últimos valores para continuar la simulación
        y0 = [datos_guardados['O2'][indice_quiebre], 
              datos_guardados['biomasa'][indice_quiebre], 
              datos_guardados['sustrato'][indice_quiebre], 
              datos_guardados['volumen'][indice_quiebre]]
        
        # Mantener los datos anteriores hasta el tiempo inicial
        for key in datos_guardados:
            datos_guardados[key] = datos_guardados[key][:indice_quiebre + 1]

    else:
        # Usar los valores ingresados por el usuario como condiciones iniciales
        y0 = [C_O2, X, S, V]

    t = np.linspace(tiempo_inicial, tiempo_final, 100)

    # Simulación
    sol = odeint(modelo, y0, t, args=(F, S_in))

    # Convertir los valores de O2 disuelto de mg/L a porcentaje de saturación
    O2_saturacion = (sol[:, 0] / C_O2_star) * 100

    # Guardar los resultados actuales
    datos_guardados['tiempo'] += t.tolist()
    datos_guardados['O2'] += O2_saturacion.tolist()
    datos_guardados['biomasa'] += sol[:, 1].tolist()
    datos_guardados['sustrato'] += sol[:, 2].tolist()
    datos_guardados['volumen'] += sol[:, 3].tolist()

    # Devolver los resultados como JSON
    resultados = {
        'tiempo': datos_guardados['tiempo'],
        'O2': datos_guardados['O2'],
        'biomasa': datos_guardados['biomasa'],
        'sustrato': datos_guardados['sustrato'],
        'volumen': datos_guardados['volumen']
    }
    return jsonify(resultados)

# Ruta para resetear los datos
@app.route('/reset', methods=['POST'])
def reset_simulation():
    global datos_guardados
    datos_guardados = {
        'tiempo': [],
        'O2': [],
        'biomasa': [],
        'sustrato': [],
        'volumen': []
    }
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    app.run(debug=True)
