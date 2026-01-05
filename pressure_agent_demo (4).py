# pressure_agent_demo.py

"""
Este módulo proporciona un agente de presión para calcular la presión óptima
en frío de los neumáticos de un camión en función de condiciones operativas
(temperatura, topografía, tipo de superficie, carga por eje, velocidad y
desgaste de neumáticos).  Además, incorpora un conjunto de
configuraciones de camiones con semirremolque y clasificaciones de
semirremolques para permitir que un sistema interactivo registre la
configuración del vehículo y la patente.

La información de configuraciones se basa en referencias sobre
configuraciones de ejes:
 - 4x2: dos ejes, un eje direccional y un eje motriz【509363212605434†L79-L83】.
 - 6x2: tres ejes con un solo eje motriz y un eje auxiliar【509363212605434†L88-L100】.
 - 6x4: tres ejes con dos ejes traseros motrices【509363212605434†L102-L108】.
 - 4x4: dos ejes, todos motrices, para vehículos todoterreno【96028779168279†L126-L131】.
 - 6x6: tres ejes con tracción en todas las ruedas【197853797804838†L136-L156】.
 - 8x2 y 8x4: cuatro ejes con doble dirección delantera; en 8x4 ambos ejes
   traseros son motrices, en 8x2 solo uno lo es【841555106129209†L30-L33】.
 - 8x6: cuatro ejes con tres ejes motrices; usado en transportes de
   palas eólicas y cargas extrapesadas【629565818139410†L88-L109】.
 - 8x8: cuatro ejes todos motrices para vehículos militares y de rescate【818121159659275†L120-L146】.
 - 10x4: cinco ejes (dos delanteros y un tridem trasero) para hormigoneras y
   volquetes【694217054497222†L45-L66】.
 - 10x6 y 10x8: cinco ejes con tres o cuatro ejes motrices, utilizados en
   aplicaciones muy pesadas【822524799009294†L320-L342】.

Nota: Las referencias en los comentarios son informativas y no se usan en
tiempo de ejecución.

Además, la normativa chilena establecida en el Decreto Supremo N.º 158/1980
señala que la combinación de un camión con uno o más remolques no puede
superar las 45 toneladas de peso bruto total (PBT)【109629155211619†L210-L233】.  La Hoja
Anexa N.º 9 al Decreto 18/93 también aclara que no se emitirá un certificado
de pesaje si el PBT excede los 45 000 kg【794604522139219†L30-L35】.  Esta versión del
agente incluye una validación para advertir al usuario cuando la suma de
las cargas por eje supere dicho límite.

Finalmente, para cuantificar el impacto energético en términos de
combustible, el agente convierte las brechas de energía (en
megajulios) a su equivalente en litros de diésel al final del flujo
interactivo.  Se utiliza para ello un valor de referencia de **35,86 MJ
por litro de diésel**, que es la densidad energética volumétrica típica
del diésel según el material educativo de la Universidad de
Waterloo【654080661782584†L62-L70】.  De este modo, el usuario puede
apreciar cuántos litros de combustible representan las diferencias de
energía calculadas.
"""

import numpy as np

# Tabla de configuraciones de camiones con semirremolque.  Cada clave es una
# cadena 'm×n' donde m es el número total de posiciones de ruedas y n el
# número de ruedas motrices.  Los valores proporcionan descripciones
# resumidas y usos habituales.
TRUCK_CONFIGS = {
    '4x2': {
        'description': 'Dos ejes: un eje delantero direccional y un eje trasero motriz',
        'typical_use': 'Transporte de carga ligera en carretera; viajes de larga distancia',
        'advantages': 'Simplicidad, bajo consumo y mantenimiento',
        'disadvantages': 'Menor tracción y capacidad de carga'
    },
    '6x2': {
        'description': 'Tres ejes: un eje motriz y un eje auxiliar (fijo, direccional o elevable)',
        'typical_use': 'Distribución y transporte medio/pesado; el eje elevable reduce consumo',
        'advantages': 'Mayor carga útil y maniobrabilidad respecto a un 4×2',
        'disadvantages': 'Menor tracción que un 6×4'
    },
    '6x4': {
        'description': 'Tres ejes: dos ejes traseros motrices (tándem) y un eje delantero',
        'typical_use': 'Cargas pesadas, construcción, uso fuera de carretera',
        'advantages': 'Alta tracción y capacidad de arrastre',
        'disadvantages': 'Mayor consumo y mantenimiento que un 6×2'
    },
    '4x4': {
        'description': 'Dos ejes, ambos motrices',
        'typical_use': 'Vehículos todoterreno, rescate, militares',
        'advantages': 'Máxima tracción',
        'disadvantages': 'Peso y consumo elevados'
    },
    '6x6': {
        'description': 'Tres ejes con tracción en todas las ruedas',
        'typical_use': 'Vehículos militares, de rescate o para remolque pesado',
        'advantages': 'Excelente movilidad en terrenos difíciles',
        'disadvantages': 'Alto consumo y coste'
    },
    '8x2': {
        'description': 'Cuatro ejes: doble dirección delantera y solo un eje trasero motriz',
        'typical_use': 'Alternativa a bitrenes en cargas de hasta 52,5 toneladas',
        'advantages': 'Estabilidad y distribución de peso mejorada',
        'disadvantages': 'Tracción limitada; requiere motor potente'
    },
    '8x4': {
        'description': 'Cuatro ejes: doble dirección delantera y dos ejes traseros motrices',
        'typical_use': 'Hormigoneras, volquetes, transporte pesado',
        'advantages': 'Gran estabilidad y alta capacidad de carga',
        'disadvantages': 'Más pesado y costoso que un 6×4'
    },
    '8x6': {
        'description': 'Cuatro ejes: tres ejes motrices (dos traseros y uno central)',
        'typical_use': 'Transportes especiales de maquinaria pesada o palas eólicas',
        'advantages': 'Tracción extrema y gran capacidad de arrastre',
        'disadvantages': 'Coste y consumo elevados'
    },
    '8x8': {
        'description': 'Cuatro ejes, todos motrices',
        'typical_use': 'Vehículos militares o de rescate de alta movilidad',
        'advantages': 'Tracción total y movilidad todoterreno',
        'disadvantages': 'Peso y complejidad muy altos'
    },
    '10x4': {
        'description': 'Cinco ejes: dos delanteros direccionales y un tridem trasero con dos ejes motrices y uno direccional',
        'typical_use': 'Hormigoneras, volquetes, camiones grúa y uso industrial',
        'advantages': 'Capacidad de carga muy alta y buena maniobrabilidad',
        'disadvantages': 'Altos costes y necesidad de motores potentes'
    },
    '10x6': {
        'description': 'Cinco ejes con tres ejes motrices (variante extrema para cargas especiales)',
        'typical_use': 'Aplicaciones de construcción y minería de alto tonelaje',
        'advantages': 'Tracción muy elevada',
        'disadvantages': 'Poco común y costoso'
    },
    '10x8': {
        'description': 'Cinco ejes con cuatro ejes motrices',
        'typical_use': 'Transporte extrapesado off‑road y vehículos especiales',
        'advantages': 'Máxima tracción y capacidad de carga',
        'disadvantages': 'Consumo, peso y mantenimiento extremadamente altos'
    },
}

# Presiones de referencia por tipo de eje para un cliente específico.
# Estas presiones se utilizan como presión de referencia (P_ref_psi) en el
# algoritmo de cálculo de presión óptima.  Los valores representan la
# presión en frío recomendada cuando el eje soporta la carga de referencia
# F_ref_kg.  Si se desea utilizar otras presiones, puede modificarse
# este diccionario.
REFERENCE_PRESSURES = {
    'steer': 110,        # presiones para ejes direccionales (delanteros)
    'tractor_other': 105, # presiones para ejes motrices o auxiliares del tractor
    'trailer': 100       # presiones para ejes del remolque
}

# Carga de referencia por eje en kilogramos.  Esta constante se utiliza
# para normalizar la presión en función de la carga real por eje.  La
# literatura sobre neumáticos comerciales indica que un neumático para
# eje direccional puede soportar cargas del orden de 2,700–3,000 kg con
# presiones de inflado en caliente de alrededor de 115 psi【412796127084028†L112-L124】.  Para simplificar el
# modelo de inflado óptimo, tomamos como referencia 6 000 kg, que es
# aproximadamente el peso por eje de un camión con una carga media.
F_REF_KG = 6000

# Límite legal de peso bruto total en Chile.
# El decreto supremo N.º 158/1980 fija que la combinación de un camión con uno
# o varios remolques no puede exceder las 45 toneladas (45 000 kg) de peso bruto
# total (PBT)【109629155211619†L210-L233】.  De igual forma, la Hoja Anexa N.º 9
# del Decreto 18/93 establece que un certificado de pesaje no puede ser
# entregado si el peso total del vehículo supera este límite【794604522139219†L30-L35】.  La
# práctica en las estaciones de pesaje es no emitir certificados a vehículos que
# excedan los 45 000 kg y solicitar descarga de la carga.
MAX_TOTAL_WEIGHT_KG = 45000


# Clasificación de semirremolques según el número de ejes
SEMITRAILER_CLASSES = {
    'S1': {'axles': 1, 'description': 'Semirremolque de un eje'},
    'S2': {'axles': 2, 'description': 'Semirremolque de dos ejes'},
    'S3': {'axles': 3, 'description': 'Semirremolque de tres ejes'},
}

# Fórmulas utilizadas en el modelo (en formato de texto).  Se incluyen aquí para
# que el usuario pueda consultarlas al final del análisis.  Las barras
# invertidas están escapadas para evitar advertencias de escape en Python.
FORMULAS = {
    'Coeficiente de resistencia al rodado':
        'c = 0.005 + b/p + c_base * p',
    'Parámetro b en función de la velocidad':
        'b = 0.01 + 0.0095 * (v/100)^2',
    'Presión óptima en caliente':
        'p_opt = sqrt(b / c_base)',
    'Energía consumida por eje':
        'E = c * W * distancia',
    'Conversión de energía a litros de diésel':
        '1 litro de diésel ≈ 35.86 MJ (energía neta)',
}

def validate_total_weight(loads_tractor, loads_trailer, max_total=MAX_TOTAL_WEIGHT_KG):
    """
    Valida que la suma de cargas por eje no supere el peso bruto total
    permitido por la normativa chilena.  En caso de exceder el límite,
    muestra un mensaje de advertencia.

    Args:
        loads_tractor (list[float]): Lista de cargas (kg) por eje del tractor.
        loads_trailer (list[float]): Lista de cargas (kg) por eje del remolque.
        max_total (float): Peso máximo permitido en kg (por defecto 45 000 kg).

    Returns:
        None
    """
    total_load = sum(loads_tractor) + sum(loads_trailer)
    if total_load > max_total:
        excess = total_load - max_total
        print(f"\nADVERTENCIA: La suma de las cargas por eje (%.1f kg) excede el límite legal de %d kg en %.1f kg." % (total_load, max_total, excess))
        print("Se recomienda reducir la carga antes de iniciar el viaje para cumplir con la legislación.")
    else:
        print(f"\nCarga total registrada: {total_load:.1f} kg (dentro del límite legal de {max_total} kg)")

def get_config_details(config_code: str):
    """Devuelve información descriptiva de la configuración de ejes.

    Args:
        config_code (str): Cadena en el formato 'm×n' (por ejemplo '6x2').

    Returns:
        dict|None: Diccionario con claves 'description', 'typical_use',
            'advantages' y 'disadvantages', o None si no se encuentra.
    """
    return TRUCK_CONFIGS.get(config_code.lower(), None)

def get_semitrailer_details(class_code: str):
    """Devuelve información sobre la clasificación del semirremolque.

    Args:
        class_code (str): Código de clase 'S1', 'S2' o 'S3'.

    Returns:
        dict|None: Diccionario con el número de ejes y descripción, o None si
            la clave no existe.
    """
    return SEMITRAILER_CLASSES.get(class_code.upper(), None)

def append_result_to_google_sheet(sheet_url: str, row: list):
    """
    Intenta agregar la fila ``row`` al final de la hoja de cálculo especificada por
    ``sheet_url``.  Este es un stub: en este entorno no tenemos acceso directo
    a las APIs de Google Sheets ni a bibliotecas como gspread por lo que no
    se realiza una inserción real.  En su lugar, la función muestra la fila
    que debería añadirse.  Para utilizar esta funcionalidad, se debe
    configurar un acceso de servicio a la API de Google Sheets y emplear
    librerías como gspread con credenciales válidas.

    Args:
        sheet_url: URL de la hoja de cálculo de Google donde se guardarán los datos.
        row: lista de valores que representan una fila de datos a guardar.
    """
    print("\n--- Registro para Google Sheets ---")
    print(f"Se debería añadir la siguiente fila a la hoja {sheet_url}:")
    print(row)

def pressure_agent(
    ambient_temp_c,
    topography_grade_percent,
    surface_type,
    load_per_axle_kg,
    speed_kmh,
    vehicle_history_factor=1.0,
    P_ref_psi=100,
    F_ref_kg=6000,
    k=1.0,
    T_hot_C=60,
    P_min_psi=70,
    P_max_psi=120
):
    """
    Calcula la presión en frío recomendada (psi) usando un modelo heurístico.

    Este modelo parte de una presión de referencia ``P_ref_psi`` que es la
    recomendación del fabricante para una carga de referencia ``F_ref_kg``. A
    partir de ahí la presión se ajusta en función de la carga real (mediante
    un exponente ``k``), la velocidad, la condición de la superficie, la
    topografía y el historial de desgaste de los neumáticos. Finalmente, se
    convierte la presión calculada en caliente a su equivalente en frío
    usando la ley de los gases ideales y se recorta a los límites
    ``P_min_psi`` y ``P_max_psi``.
    """
    # Temperaturas absolutas para conversión gas ideal
    T_cold_K = ambient_temp_c + 273.15
    T_hot_K = T_hot_C + 273.15
    # Ajuste por carga (relación con la carga de referencia)
    load_ratio = load_per_axle_kg / F_ref_kg
    P_hot_psi = P_ref_psi * (load_ratio ** k)
    # Ajustes por velocidad
    if speed_kmh > 90:
        P_hot_psi *= 1.05
    # Ajustes por superficie
    surface = surface_type.lower()
    if surface in ['gravel', 'sand', 'rough']:
        P_hot_psi *= 0.9
    elif surface in ['wet']:
        P_hot_psi *= 0.95
    # Ajustes por topografía
    if topography_grade_percent > 5:
        P_hot_psi *= 0.95
    # Ajuste por historial de neumático
    P_hot_psi *= vehicle_history_factor
    # Conversión a presión en frío con ley de los gases ideales
    P_cold_psi = P_hot_psi * (T_cold_K / T_hot_K)
    # Respeto de límites mínimos y máximos
    P_cold_psi = max(min(P_cold_psi, P_max_psi), P_min_psi)
    return P_cold_psi


# --- Nuevo modelo físico para presión óptima ---
def pressure_optimum_scientific(speed_kmh: float, P_ref_psi: float, baseline_speed_kmh: float = 80.0) -> float:
    r"""
    Calcula la presión de inflado en caliente que minimiza la resistencia al
    rodado utilizando un modelo físico de rolling resistance.

    Este modelo considera la fórmula semiempírica para el coeficiente de
    resistencia al rodado en carreteras secas:

        c = 0.005 + \frac{b}{p} + c_\text{base} p,

    donde p es la presión en bar, v la velocidad en km/h y b se define como

        b = 0.01 + 0.0095\left(\frac{v}{100}\right)^2【751715621022178†L74-L87】.

    Para calibrar el modelo se fija que a una velocidad de referencia
    ``baseline_speed_kmh`` la presión óptima coincide con la presión de
    referencia ``P_ref_psi``. A partir de ahí se calcula la constante
    ``c_base`` que equilibra el término lineal de alta presión y se utiliza
    para obtener la presión óptima a la velocidad real.

    Args:
        speed_kmh: Velocidad media del vehículo en km/h.
        P_ref_psi: Presión de referencia en psi (correspondiente a la presión
            deseada en caliente a la velocidad de referencia).
        baseline_speed_kmh: Velocidad utilizada para calibrar el modelo (por
            defecto 80 km/h).

    Returns:
        float: presión óptima en caliente (psi) que minimiza la resistencia
            al rodado en las condiciones de velocidad dadas.
    """
    # Constante b para la velocidad de referencia
    b0 = 0.01 + 0.0095 * ((baseline_speed_kmh) / 100.0) ** 2
    # Convertir P_ref de psi a bar
    P_ref_bar = P_ref_psi * 0.0689476
    # Calcular c_base de manera que el óptimo a la velocidad de referencia sea P_ref
    c_base = b0 / (P_ref_bar ** 2)
    # Calcular b para la velocidad real
    b = 0.01 + 0.0095 * ((speed_kmh) / 100.0) ** 2
    # Presión óptima en bar según el modelo c = a + b/p + c_base p
    P_opt_bar = (b / c_base) ** 0.5
    # Convertir a psi
    P_opt_psi = P_opt_bar / 0.0689476
    return P_opt_psi

def compute_rolling_coefficient(psi: float, speed_kmh: float) -> float:
    """Calcula el coeficiente de resistencia al rodado para un neumático.

    Usa la fórmula empírica derivada de la literatura en la que el coeficiente
    depende de la presión (en bar) y de la velocidad de circulación:

        c = 0.005 + (0.01 + 0.0095*(v/100)**2)/p

    donde p es la presión en bar y v la velocidad en km/h【66302365848204†L74-L87】.

    Args:
        psi: presión en frío (en psi).
        speed_kmh: velocidad media de la ruta en km/h.

    Returns:
        float: el coeficiente c adimensional.
    """
    # Conversión de psi a bar: 1 psi = 0.0689476 bar
    p_bar = psi * 0.0689476
    # Fórmula del coeficiente de resistencia al rodado
    c = 0.005 + (0.01 + 0.0095 * (speed_kmh / 100.0) ** 2) / p_bar
    return c


def compute_energy_consumption(psi: float, speed_kmh: float, load_per_axle_kg: float,
                               distance_km: float) -> float:
    """Calcula la energía (en julios) consumida por un neumático en un viaje.

    Se calcula la fuerza de rodado F_r = c * W, siendo c el coeficiente de
    rodado y W el peso (masa*gravedad).  La energía por unidad de longitud es
    F_r, de modo que la energía total para una distancia d es F_r * d.

    Args:
        psi: presión de inflado en frío en psi.
        speed_kmh: velocidad media en km/h.
        load_per_axle_kg: carga en kg soportada por el eje asociado al neumático.
        distance_km: distancia recorrida en km.

    Returns:
        float: energía consumida en julios (J) para ese neumático.
    """
    c = compute_rolling_coefficient(psi, speed_kmh)
    weight_newtons = load_per_axle_kg * 9.81
    # Fuerza de rodadura (N)
    F_r = c * weight_newtons
    # Convertir distancia a metros
    distance_m = distance_km * 1000.0
    # Energía (J) = fuerza * distancia
    energy_j = F_r * distance_m
    return energy_j


def run_interactive_agent():
    """
    Ejecuta un flujo interactivo completo para calcular y comparar
    la presión óptima y la energía consumida por eje, diferenciando entre
    ejes direccionales (steer), ejes motrices/auxiliares del tractor y
    ejes del semirremolque. Para cada tipo se utiliza una presión de
    referencia distinta definida en ``REFERENCE_PRESSURES``.

    **Importante**: a partir de esta versión no se solicita ni se incorpora
    el estado de desgaste de los neumáticos. Se asume que todos los
    neumáticos están usados y se aplica un pequeño factor de reducción
    interno al cálculo de la presión óptima.

    Pasos:
    1. Solicita datos básicos del vehículo: configuración del tractor,
       patente y clasificación del semirremolque.
    2. Solicita variables operativas del viaje (temperatura ambiente,
       topografía, superficie, velocidad media y distancia del viaje).
    3. Determina el número de ejes del tractor a partir de la configuración
       y el número de ejes direccionales (steer) según un mapeo estándar.
       Calcula el número de ejes del remolque a partir de la clase S1/S2/S3.
    4. Solicita la carga en frío para cada eje del tractor y del remolque
       individualmente.
    5. Calcula la presión óptima en frío para cada eje utilizando la
       referencia apropiada: 110 psi para ejes direccionales, 105 psi
       para otros ejes del tractor y 100 psi para ejes del remolque.
       La presión óptima se ajusta según la carga y se aplica un factor
       constante (0,98) que representa un desgaste típico, ya que se
       asume que todos los neumáticos están usados.
    6. Construye el gemelo digital: estima el consumo de energía por eje
       y total usando las presiones óptimas.
    7. Solicita las presiones reales en frío para cada eje y calcula
       el consumo energético real.
    8. Compara los consumos estimado y real, mostrando la brecha por eje
       y total.
    9. Presenta una tabla de equivalencia entre las brechas de energía
       (en megajulios) y el volumen de diésel que representaría esa
       energía, utilizando una densidad energética típica de 35,86 MJ/L
       para el diésel【654080661782584†L62-L70】.
    """
    # 1. Datos básicos del vehículo
    config_code = input("Ingrese la configuración del tractor (ej. '6x4'): ").strip().lower()
    plate = input("Ingrese la patente del vehículo: ").strip()
    semi_class = input("Ingrese la clasificación del semirremolque (S1, S2 o S3): ").strip().upper()

    config_info = get_config_details(config_code)
    semi_info = get_semitrailer_details(semi_class)

    print("\n--- Información del vehículo ---")
    if config_info:
        print(f"Configuración {config_code}: {config_info['description']}")
        print(f"Uso típico: {config_info['typical_use']}")
        print(f"Ventajas: {config_info['advantages']}")
        print(f"Desventajas: {config_info['disadvantages']}")
    else:
        print(f"Advertencia: configuración '{config_code}' no registrada.")

    # Mostrar información del semirremolque
    if semi_info:
        print(f"Semirremolque {semi_class}: {semi_info['description']} (ejes: {semi_info['axles']})")
    else:
        print(f"Advertencia: clasificación de semirremolque '{semi_class}' no reconocida.")

    # 2. Variables operativas del viaje
    print("\n--- Datos del viaje ---")
    try:
        distance_km = float(input("Ingrese la distancia del viaje (km): ").strip())
    except ValueError:
        distance_km = 1.0
    try:
        ambient_temp_c = float(input("Temperatura ambiente (°C): ").strip())
    except ValueError:
        ambient_temp_c = 20.0
    try:
        topography_grade_percent = float(input("Pendiente promedio de la ruta (%): ").strip())
    except ValueError:
        topography_grade_percent = 0.0
    surface_type = input("Tipo de superficie (asfalto, mojado, gravilla, arena, etc.): ").strip().lower()
    try:
        speed_kmh = float(input("Velocidad promedio (km/h): ").strip())
    except ValueError:
        speed_kmh = 80.0
    # Fijar un factor de desgaste para neumáticos usados.  Se asume
    # que todos los neumáticos están usados, por lo que aplicamos una
    # reducción uniforme del 2 % en la presión óptima.  Si se desea
    # cambiar este valor, modifique la constante aquí.
    vehicle_history_factor = 0.98

    # 3. Número de ejes del tractor (posiciones/2) y del remolque
    try:
        positions = int(config_code.split('x')[0])
        num_tractor_axles = positions // 2
    except (ValueError, IndexError):
        num_tractor_axles = None

    if num_tractor_axles is None or num_tractor_axles <= 0:
        try:
            num_tractor_axles = int(input("No se pudo inferir el número de ejes del tractor. Ingrese la cantidad de ejes: ").strip())
        except ValueError:
            num_tractor_axles = 2

    # Número de ejes del remolque según la clase
    if semi_info and 'axles' in semi_info:
        num_trailer_axles = semi_info['axles']
    else:
        try:
            num_trailer_axles = int(input("Ingrese el número de ejes del semirremolque: ").strip())
        except ValueError:
            num_trailer_axles = 1

    # 3b. Determinar cuántos ejes son direccionales (steer) en el tractor
    # Mapeo estándar basado en configuraciones conocidas; se puede ampliar si se añaden nuevas
    steer_axles_mapping = {
        '4x2': 1,
        '6x2': 1,
        '6x4': 1,
        '4x4': 1,
        '6x6': 1,
        '8x2': 2,
        '8x4': 2,
        '8x6': 2,
        '8x8': 2,
        '10x4': 2,
        '10x6': 2,
        '10x8': 2,
    }
    num_steer_axles = steer_axles_mapping.get(config_code, None)
    if num_steer_axles is None:
        # Si la configuración no está en el mapeo, preguntamos al usuario
        try:
            num_steer_axles = int(input(f"Ingrese el número de ejes direccionales del tractor para la configuración '{config_code}': ").strip())
        except ValueError:
            num_steer_axles = 1

    print("\n--- Carga por eje ---")
    loads_tractor = []
    loads_trailer = []
    # Cargas por eje del tractor
    for i in range(num_tractor_axles):
        while True:
            try:
                load_i = float(input(f"Carga en frío del eje del tractor {i+1} (kg): ").strip())
                loads_tractor.append(load_i)
                break
            except ValueError:
                print("Entrada no válida. Por favor, introduzca un número.")
    # Cargas por eje del remolque
    for i in range(num_trailer_axles):
        while True:
            try:
                load_i = float(input(f"Carga en frío del eje del remolque {i+1} (kg): ").strip())
                loads_trailer.append(load_i)
                break
            except ValueError:
                print("Entrada no válida. Por favor, introduzca un número.")

    # Validar que la suma de cargas no exceda el límite de 45 000 kg
    validate_total_weight(loads_tractor, loads_trailer)

    # 4. Calcular la presión óptima para cada eje usando el modelo científico
    optimal_pressures_tractor = []
    for i, load_i in enumerate(loads_tractor):
        # Seleccionar la presión de referencia según si el eje es direccional o no
        if i < num_steer_axles:
            P_ref = REFERENCE_PRESSURES['steer']
        else:
            P_ref = REFERENCE_PRESSURES['tractor_other']
        # Presión óptima en caliente basada en modelo físico
        P_hot_opt_psi = pressure_optimum_scientific(speed_kmh=speed_kmh, P_ref_psi=P_ref)
        # Ajuste por carga: aumentar la presión en proporción a la relación de carga respecto a la carga de referencia.
        # Para cargas mayores que la referencia, la presión se incrementa; para cargas menores, disminuye.
        load_ratio = load_i / F_REF_KG
        k = 1.0  # exponente lineal; puede ajustarse según datos empíricos
        P_hot_opt_psi *= load_ratio ** k
        # Ajustes por superficie y topografía
        surface = surface_type.lower()
        if surface in ['gravel', 'sand', 'rough']:
            P_hot_opt_psi *= 0.9
        elif surface in ['wet']:
            P_hot_opt_psi *= 0.95
        if topography_grade_percent > 5:
            P_hot_opt_psi *= 0.95
        # Ajuste por desgaste
        P_hot_opt_psi *= vehicle_history_factor
        # Convertir a presión en frío usando ley de los gases ideales
        T_hot_K = (60) + 273.15
        T_cold_K = ambient_temp_c + 273.15
        P_cold = P_hot_opt_psi * (T_cold_K / T_hot_K)
        # Establecer un mínimo razonable (10 psi por debajo de la referencia) para evitar valores muy bajos
        min_psi = max(P_ref - 10, 70)
        # Limitar entre mínimo y máximo
        P_cold = max(min(P_cold, 120), min_psi)
        optimal_pressures_tractor.append(P_cold)
    optimal_pressures_trailer = []
    for load_i in loads_trailer:
        P_ref = REFERENCE_PRESSURES['trailer']
        # Presión óptima en caliente basada en modelo físico
        P_hot_opt_psi = pressure_optimum_scientific(speed_kmh=speed_kmh, P_ref_psi=P_ref)
        # Ajuste por carga del remolque
        load_ratio = load_i / F_REF_KG
        k = 1.0
        P_hot_opt_psi *= load_ratio ** k
        # Ajustes por superficie y topografía
        surface = surface_type.lower()
        if surface in ['gravel', 'sand', 'rough']:
            P_hot_opt_psi *= 0.9
        elif surface in ['wet']:
            P_hot_opt_psi *= 0.95
        if topography_grade_percent > 5:
            P_hot_opt_psi *= 0.95
        # Ajuste por desgaste
        P_hot_opt_psi *= vehicle_history_factor
        # Conversión a presión en frío
        T_hot_K = (60) + 273.15
        T_cold_K = ambient_temp_c + 273.15
        P_cold = P_hot_opt_psi * (T_cold_K / T_hot_K)
        # Mínimo razonable para remolque
        min_psi = max(P_ref - 10, 70)
        P_cold = max(min(P_cold, 120), min_psi)
        optimal_pressures_trailer.append(P_cold)

    # 5. Energía óptima por eje (gemelo digital)
    energies_opt_tractor = []
    for i, p_i in enumerate(optimal_pressures_tractor):
        energy_i = compute_energy_consumption(
            psi=p_i,
            speed_kmh=speed_kmh,
            load_per_axle_kg=loads_tractor[i],
            distance_km=distance_km
        )
        energies_opt_tractor.append(energy_i)
    energies_opt_trailer = []
    for i, p_i in enumerate(optimal_pressures_trailer):
        energy_i = compute_energy_consumption(
            psi=p_i,
            speed_kmh=speed_kmh,
            load_per_axle_kg=loads_trailer[i],
            distance_km=distance_km
        )
        energies_opt_trailer.append(energy_i)
    energy_total_opt = sum(energies_opt_tractor) + sum(energies_opt_trailer)

    print("\n--- Gemelo digital (configuración óptima) ---")
    # Mostrar presiones óptimas del tractor
    for i, (p_i, e_i) in enumerate(zip(optimal_pressures_tractor, energies_opt_tractor)):
        tipo = 'direccional' if i < num_steer_axles else 'tractor'
        print(f"Tractor eje {i+1} ({tipo}): presión óptima {p_i:.1f} psi, energía estimada {e_i/1e6:.3f} MJ")
    # Mostrar presiones óptimas del remolque
    for i, (p_i, e_i) in enumerate(zip(optimal_pressures_trailer, energies_opt_trailer)):
        print(f"Remolque eje {i+1}: presión óptima {p_i:.1f} psi, energía estimada {e_i/1e6:.3f} MJ")
    print(f"Energía total estimada (todos los ejes): {energy_total_opt/1e6:.3f} MJ")

    # 6. Solicitar presiones reales por eje
    print("\n--- Introducir presiones reales en frío ---")
    actual_pressures_tractor = []
    for i in range(num_tractor_axles):
        while True:
            try:
                p = float(input(f"Presión real del eje del tractor {i+1} (psi): ").strip())
                actual_pressures_tractor.append(p)
                break
            except ValueError:
                print("Entrada no válida. Por favor, introduzca un número.")
    actual_pressures_trailer = []
    for i in range(num_trailer_axles):
        while True:
            try:
                p = float(input(f"Presión real del eje del remolque {i+1} (psi): ").strip())
                actual_pressures_trailer.append(p)
                break
            except ValueError:
                print("Entrada no válida. Por favor, introduzca un número.")

    # 7. Calcular energía real para cada eje
    energies_real_tractor = []
    for i, p in enumerate(actual_pressures_tractor):
        energy_real_i = compute_energy_consumption(
            psi=p,
            speed_kmh=speed_kmh,
            load_per_axle_kg=loads_tractor[i],
            distance_km=distance_km
        )
        energies_real_tractor.append(energy_real_i)
    energies_real_trailer = []
    for i, p in enumerate(actual_pressures_trailer):
        energy_real_i = compute_energy_consumption(
            psi=p,
            speed_kmh=speed_kmh,
            load_per_axle_kg=loads_trailer[i],
            distance_km=distance_km
        )
        energies_real_trailer.append(energy_real_i)
    energy_total_real = sum(energies_real_tractor) + sum(energies_real_trailer)

    # 8. Comparación
    print("\n--- Comparación de consumos ---")
    # Tractor
    for i, (opt, real) in enumerate(zip(energies_opt_tractor, energies_real_tractor)):
        gap = real - opt
        pct = (gap / opt) * 100.0 if opt > 0 else 0.0
        sign = "más" if gap > 0 else "menos"
        tipo = 'direccional' if i < num_steer_axles else 'tractor'
        print(f"Tractor eje {i+1} ({tipo}): energía real {real/1e6:.3f} MJ, brecha {abs(gap)/1e6:.3f} MJ {sign} (" +
              f"{abs(pct):.2f}% {'más' if gap > 0 else 'menos'} energía)")
    # Remolque
    for i, (opt, real) in enumerate(zip(energies_opt_trailer, energies_real_trailer)):
        gap = real - opt
        pct = (gap / opt) * 100.0 if opt > 0 else 0.0
        sign = "más" if gap > 0 else "menos"
        print(f"Remolque eje {i+1}: energía real {real/1e6:.3f} MJ, brecha {abs(gap)/1e6:.3f} MJ {sign} (" +
              f"{abs(pct):.2f}% {'más' if gap > 0 else 'menos'} energía)")
    # Total
    energy_gap_total = energy_total_real - energy_total_opt
    percent_gap_total = (energy_gap_total / energy_total_opt) * 100.0 if energy_total_opt > 0 else 0.0
    sign_total = "más" if energy_gap_total > 0 else "menos"
    print(f"\nEnergía total real para el viaje: {energy_total_real/1e6:.3f} MJ")
    print(f"Diferencia total respecto al gemelo digital: {abs(energy_gap_total)/1e6:.3f} MJ {sign_total} (" +
          f"{abs(percent_gap_total):.2f}% {'más' if energy_gap_total > 0 else 'menos'} energía)")

    # 9. Tabla de equivalencia MJ → litros de diésel
    # Densidad energética típica del diésel (MJ por litro)【654080661782584†L62-L70】
    MJ_PER_LITER_DIESEL = 35.86
    print("\n--- Equivalencia de brechas de energía a litros de diésel ---")
    # Para cada brecha de energía, convertir primero a MJ y luego a litros
    # L = (|gap_J| / 1e6) / MJ_PER_LITER_DIESEL
    for i, (opt, real) in enumerate(zip(energies_opt_tractor, energies_real_tractor)):
        gap_j = real - opt  # diferencia en julios
        gap_mj = abs(gap_j) / 1e6
        liters = gap_mj / MJ_PER_LITER_DIESEL
        tipo = 'direccional' if i < num_steer_axles else 'tractor'
        print(f"Tractor eje {i+1} ({tipo}): {gap_mj:.3f} MJ -> {liters:.3f} L de diésel")
    for i, (opt, real) in enumerate(zip(energies_opt_trailer, energies_real_trailer)):
        gap_j = real - opt
        gap_mj = abs(gap_j) / 1e6
        liters = gap_mj / MJ_PER_LITER_DIESEL
        print(f"Remolque eje {i+1}: {gap_mj:.3f} MJ -> {liters:.3f} L de diésel")
    total_gap_mj = abs(energy_gap_total) / 1e6
    total_liters = total_gap_mj / MJ_PER_LITER_DIESEL
    print(f"Total: {total_gap_mj:.3f} MJ -> {total_liters:.3f} L de diésel")

    # 10. Mostrar fórmulas utilizadas
    print("\n--- Fórmulas utilizadas ---")
    for nombre, formula in FORMULAS.items():
        print(f"{nombre}: {formula}")

    # 11. Preparar y guardar registro en Google Sheets (stub)
    from datetime import datetime
    fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Recopilar datos en una lista. Algunos campos son cadenas concatenadas por ';' para
    # almacenar múltiples valores en una celda.
    row_data = [
        fecha_hora,
        plate,
        config_code,
        semi_class,
        distance_km,
        speed_kmh,
        ';'.join(f"{x:.1f}" for x in loads_tractor + loads_trailer),
        ';'.join(f"{x:.1f}" for x in optimal_pressures_tractor + optimal_pressures_trailer),
        ';'.join(f"{x:.1f}" for x in actual_pressures_tractor + actual_pressures_trailer),
        ';'.join(f"{(real - opt)/1e6:.3f}" for opt, real in list(zip(energies_opt_tractor + energies_opt_trailer, energies_real_tractor + energies_real_trailer))),
        total_gap_mj,
        total_liters
    ]
    SHEET_URL = "https://docs.google.com/spreadsheets/d/1UFSDPrg_PSLB7RAAhqgsS06ZzVJ2hXMB_tox88c122A/edit?usp=sharing"
    append_result_to_google_sheet(SHEET_URL, row_data)


if __name__ == '__main__':
    run_interactive_agent()
