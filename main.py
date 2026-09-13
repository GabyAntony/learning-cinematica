import dataclasses
import json
import math
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sympy as sp


# ============================================================
# CONFIGURACIÓN
# ============================================================

G = 9.81


# ============================================================
# ECUACIONES
# ============================================================

@dataclasses.dataclass(frozen=True)
class Ecuaciones:
    """
    Catálogo central de ecuaciones de cinemática.
    Las ecuaciones son simbólicas y se pueden usar con SymPy.
    """

    # -------------------------
    # MRU
    # -------------------------
    x0, x, v, t = sp.symbols("x0 x v t", real=True)

    MRU_x = sp.Eq(x, x0 + v * t)
    MRU_v = sp.Eq(v, (x - x0) / t)
    MRU_t = sp.Eq(t, (x - x0) / v)

    # -------------------------
    # MRUA
    # -------------------------
    v0, vf, a = sp.symbols("v0 vf a", real=True)

    MRUA_vf = sp.Eq(vf, v0 + a * t)

    MRUA_dx = sp.Eq(
        x - x0,
        v0 * t + sp.Rational(1, 2) * a * t**2
    )

    MRUA_sin_t = sp.Eq(
        vf**2,
        v0**2 + 2 * a * (x - x0)
    )

    MRUA_a = sp.Eq(
        a,
        (vf - v0) / t
    )

    vmedia = sp.symbols("vmedia", real=True)

    MRUA_vmedia = sp.Eq(
        vmedia,
        (v0 + vf) / 2
    )

    MRUA_dx_vmedia = sp.Eq(
        x - x0,
        ((v0 + vf) / 2) * t
    )

    # -------------------------
    # PARABÓLICO
    # -------------------------
    theta = sp.symbols("theta", real=True)
    g, y0 = sp.symbols("g y0", positive=True, real=True)

    v0x, v0y = sp.symbols("v0x v0y", real=True)
    y, vx, vy = sp.symbols("y vx vy", real=True)

    PAR_v0x = sp.Eq(
        v0x,
        v0 * sp.cos(theta)
    )

    PAR_v0y = sp.Eq(
        v0y,
        v0 * sp.sin(theta)
    )

    PAR_x = sp.Eq(
        x,
        x0 + v0x * t
    )

    PAR_y = sp.Eq(
        y,
        y0 + v0y * t - sp.Rational(1, 2) * g * t**2
    )

    PAR_vy = sp.Eq(
        vy,
        v0y - g * t
    )

    tmax, hmax = sp.symbols("tmax hmax", real=True)

    PAR_tmax = sp.Eq(
        tmax,
        v0y / g
    )

    PAR_hmax = sp.Eq(
        hmax,
        y0 + v0y**2 / (2 * g)
    )

    # -------------------------
    # CIRCULAR
    # -------------------------
    r, omega, T, f = sp.symbols(
        "r omega T f",
        positive=True,
        real=True
    )

    vc, ac = sp.symbols("vc ac", positive=True, real=True)
    alpha = sp.symbols("alpha", real=True)
    omega0, omegaf = sp.symbols("omega0 omegaf", real=True)
    theta0, thetaf = sp.symbols("theta0 thetaf", real=True)

    CIRC_v = sp.Eq(
        vc,
        omega * r
    )

    CIRC_T = sp.Eq(
        T,
        2 * sp.pi / omega
    )

    # Ecuación inversa para calcular T desde omega (misma que arriba pero para el catálogo)
    CIRC_T_omega = sp.Eq(
        T,
        2 * sp.pi / omega
    )

    CIRC_f = sp.Eq(
        f,
        1 / T
    )

    CIRC_omega_f = sp.Eq(
        omega,
        2 * sp.pi * f
    )

    CIRC_ac = sp.Eq(
        ac,
        vc**2 / r
    )

    CIRC_ac_omega = sp.Eq(
        ac,
        omega**2 * r
    )

    # Movimiento circular uniformemente acelerado
    CIRC_omega_final = sp.Eq(
        omegaf,
        omega0 + alpha * t
    )

    CIRC_theta = sp.Eq(
        thetaf,
        theta0 + omega0 * t + sp.Rational(1, 2) * alpha * t**2
    )

    CIRC_omega_sin_t = sp.Eq(
        omegaf**2,
        omega0**2 + 2 * alpha * (thetaf - theta0)
    )

    CIRC_at = sp.Eq(
        sp.Symbol("at"),
        alpha * r
    )


# ============================================================
# UTILIDADES
# ============================================================

def numero(valor: Any) -> float:
    """Convierte un resultado numérico de Python/SymPy a float."""
    return float(sp.N(valor))


def mostrar_numero(valor: float, cifras: int = 6) -> str:
    """Formatea números para la interfaz de consola."""
    if abs(valor) < 1e-10:
        valor = 0.0

    texto = f"{valor:.{cifras}f}".rstrip("0").rstrip(".")
    return texto if texto else "0"


def leer_float(
    mensaje: str,
    permitir_negativo: bool = True,
    permitir_cero: bool = True
) -> float:
    """Lee un número real desde consola."""
    while True:
        try:
            valor = float(input(mensaje).strip())

            if not permitir_negativo and valor < 0:
                print("El valor no puede ser negativo.")
                continue

            if not permitir_cero and abs(valor) < 1e-12:
                print("El valor no puede ser cero.")
                continue

            return valor

        except ValueError:
            print("Ingresa un número válido.")


def leer_entero(mensaje: str, minimo: int = 0) -> int:
    while True:
        try:
            valor = int(input(mensaje).strip())

            if valor < minimo:
                print(f"Ingresa un entero mayor o igual que {minimo}.")
                continue

            return valor

        except ValueError:
            print("Ingresa un número entero válido.")


def grados_a_radianes(grados: float) -> float:
    return math.radians(grados)


def radianes_a_grados(radianes: float) -> float:
    return math.degrees(radianes)


def resolver_ecuacion(
    ecuacion: sp.Equality,
    variable: sp.Symbol,
    datos: Dict[sp.Symbol, Any]
) -> List[sp.Expr]:
    """
    Sustituye datos conocidos y utiliza SymPy para despejar
    la variable solicitada.
    """
    ecuacion_sustituida = ecuacion.subs(datos)
    return sp.solve(ecuacion_sustituida, variable)


def soluciones_reales(soluciones: List[sp.Expr]) -> List[float]:
    """Devuelve únicamente soluciones reales numéricas."""
    resultado = []

    for solucion in soluciones:
        try:
            valor = complex(sp.N(solucion))

            if abs(valor.imag) < 1e-9:
                resultado.append(float(valor.real))

        except (TypeError, ValueError):
            pass

    return resultado


# ============================================================
# PARSER DE ENTRADA FLEXIBLE
# ============================================================

class ParserInput:
    """
    Parser flexible para extraer variables de texto en lenguaje natural.
    Soporta formatos como: "x0 = 10m, t= 5.5s, v = 20", "calcula w con v=15 y r=3"
    """

    # Mapeo de alias de variables a símbolos de SymPy
    ALIAS_VARIABLES = {
        # Posición
        'x0': 'x0', 'x_inicial': 'x0', 'posicion_inicial': 'x0',
        'x': 'x', 'posicion': 'x', 'posicion_final': 'x',
        'dx': 'x', 'desplazamiento': 'x',
        # Velocidad
        'v': 'v', 'velocidad': 'v',
        'v0': 'v0', 'v_inicial': 'v0', 'velocidad_inicial': 'v0',
        'vf': 'vf', 'v_final': 'vf', 'velocidad_final': 'vf',
        'vx': 'vx', 'velocidad_x': 'vx',
        'vy': 'vy', 'velocidad_y': 'vy',
        'v0x': 'v0x', 'v0y': 'v0y',
        'vmedia': 'vmedia', 'velocidad_media': 'vmedia',
        'vc': 'vc', 'velocidad_tangencial': 'vc',
        # Tiempo
        't': 't', 'tiempo': 't',
        'tmax': 'tmax', 'tiempo_max': 'tmax',
        # Aceleración
        'a': 'a', 'aceleracion': 'a',
        'ac': 'ac', 'aceleracion_centrípeta': 'ac',
        'at': 'at', 'aceleracion_tangencial': 'at',
        # Movimiento circular
        'r': 'r', 'radio': 'r',
        'omega': 'omega', 'w': 'omega', 'velocidad_angular': 'omega',
        'omega0': 'omega0', 'w0': 'omega0',
        'omegaf': 'omegaf', 'wf': 'omegaf',
        'alpha': 'alpha', 'aceleracion_angular': 'alpha',
        'T': 'T', 'periodo': 'T',
        'f': 'f', 'frecuencia': 'f',
        # Movimiento parabólico
        'theta': 'theta', 'angulo': 'theta', 'angle': 'theta',
        'g': 'g', 'gravedad': 'g',
        'y0': 'y0', 'y_inicial': 'y0', 'altura_inicial': 'y0',
        'y': 'y', 'altura': 'y',
        'hmax': 'hmax', 'altura_max': 'hmax',
        # Otros
        'theta0': 'theta0', 'theta_inicial': 'theta0',
        'thetaf': 'thetaf', 'theta_final': 'thetaf',
    }

    # Patrones regex para extraer valores
    PATRON_VALOR = re.compile(
        r"""
        ([a-zA-Z_][a-zA-Z0-9_]*)      # Nombre de variable
        \s*=\s*                       # Igual con espacios opcionales
        ([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)  # Número (incluye notación científica)
        """,
        re.VERBOSE
    )

    PATRON_CALCULAR = re.compile(
        r'(?:calcula|calcular|encuentra|encuentre|determina|determine|hallar|hallar|resolver|resuelve)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        re.IGNORECASE
    )

    @classmethod
    def parsear_entrada(cls, texto: str) -> Tuple[Dict[str, float], Optional[str]]:
        """
        Parsea texto de entrada y extrae variables conocidas y variable objetivo.

        Args:
            texto: Texto de entrada del usuario

        Returns:
            Tuple con (diccionario de variables, variable objetivo a calcular)
        """
        variables = {}
        objetivo = None

        # Normalizar texto
        texto = texto.lower().strip()

        # Extraer variable objetivo si existe
        match_objetivo = cls.PATRON_CALCULAR.search(texto)
        if match_objetivo:
            objetivo_raw = match_objetivo.group(1)
            objetivo = cls._normalizar_variable(objetivo_raw)

        # Extraer pares clave-valor
        for match in cls.PATRON_VALOR.finditer(texto):
            var_raw = match.group(1)
            valor_str = match.group(2)

            var_normalizada = cls._normalizar_variable(var_raw)
            if var_normalizada:
                try:
                    valor = float(valor_str)
                    variables[var_normalizada] = valor
                except ValueError:
                    continue

        return variables, objetivo

    @classmethod
    def _normalizar_variable(cls, var: str) -> Optional[str]:
        """Normaliza un nombre de variable usando el mapeo de alias."""
        var_lower = var.lower().strip()
        return cls.ALIAS_VARIABLES.get(var_lower, var_lower)

    @classmethod
    def obtener_simbolo(cls, nombre: str) -> Optional[sp.Symbol]:
        """
        Obtiene el símbolo de SymPy correspondiente a un nombre de variable.
        """
        nombre_normalizado = cls._normalizar_variable(nombre)

        # Buscar en los símbolos de Ecuaciones
        for attr_name in dir(Ecuaciones):
            attr = getattr(Ecuaciones, attr_name)
            if isinstance(attr, sp.Symbol) and attr.name == nombre_normalizado:
                return attr

        return None


# ============================================================
# RESOLVEDOR DINÁMICO CON SYMPY
# ============================================================

class DynamicSolver:
    """
    Resolvedor universal que usa el catálogo de ecuaciones de SymPy
    para calcular variables dinámicamente.
    """

    # Catálogo de ecuaciones organizado por tema
    ECUACIONES_POR_TEMA = {
        'MRU': [
            (Ecuaciones.MRU_x, Ecuaciones.x),
            (Ecuaciones.MRU_v, Ecuaciones.v),
            (Ecuaciones.MRU_t, Ecuaciones.t),
        ],
        'MRUA': [
            (Ecuaciones.MRUA_vf, Ecuaciones.vf),
            (Ecuaciones.MRUA_dx, Ecuaciones.x),
            (Ecuaciones.MRUA_a, Ecuaciones.a),
            (Ecuaciones.MRUA_vmedia, Ecuaciones.vmedia),
            (Ecuaciones.MRUA_dx_vmedia, Ecuaciones.x),
            (Ecuaciones.MRUA_sin_t, Ecuaciones.vf),
        ],
        'Parabólico': [
            (Ecuaciones.PAR_v0x, Ecuaciones.v0x),
            (Ecuaciones.PAR_v0y, Ecuaciones.v0y),
            (Ecuaciones.PAR_x, Ecuaciones.x),
            (Ecuaciones.PAR_y, Ecuaciones.y),
            (Ecuaciones.PAR_vy, Ecuaciones.vy),
            (Ecuaciones.PAR_tmax, Ecuaciones.tmax),
            (Ecuaciones.PAR_hmax, Ecuaciones.hmax),
        ],
        'Circular': [
            (Ecuaciones.CIRC_v, Ecuaciones.vc),
            (Ecuaciones.CIRC_T, Ecuaciones.T),
            (Ecuaciones.CIRC_T_omega, Ecuaciones.T),  # Para calcular T desde omega
            (Ecuaciones.CIRC_f, Ecuaciones.f),
            (Ecuaciones.CIRC_omega_f, Ecuaciones.omega),
            (Ecuaciones.CIRC_omega_f, Ecuaciones.f),  # Para calcular f desde omega
            (Ecuaciones.CIRC_ac, Ecuaciones.ac),
            (Ecuaciones.CIRC_ac_omega, Ecuaciones.ac),
            (Ecuaciones.CIRC_omega_final, Ecuaciones.omegaf),
            (Ecuaciones.CIRC_theta, Ecuaciones.thetaf),
            (Ecuaciones.CIRC_omega_sin_t, Ecuaciones.omegaf),
            (Ecuaciones.CIRC_at, sp.Symbol("at")),
        ],
    }

    @classmethod
    def resolver(
        cls,
        datos_conocidos: Dict[str, float],
        variable_objetivo: str,
        tema: Optional[str] = None
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Resuelve una variable usando el catálogo de ecuaciones.

        Args:
            datos_conocidos: Diccionario con variables conocidas {nombre: valor}
            variable_objetivo: Nombre de la variable a calcular
            tema: Tema específico (opcional, si None busca en todos)

        Returns:
            Tuple con (resultado calculado, mensaje de error si falla)
        """
        # Obtener símbolo objetivo
        simbolo_objetivo = ParserInput.obtener_simbolo(variable_objetivo)
        if not simbolo_objetivo:
            return None, f"Variable '{variable_objetivo}' no reconocida"

        # Convertir datos a símbolos de SymPy
        datos_sym = {}
        for nombre, valor in datos_conocidos.items():
            simbolo = ParserInput.obtener_simbolo(nombre)
            if simbolo:
                datos_sym[simbolo] = valor

        # Determinar qué ecuaciones probar
        ecuaciones_a_probar = []
        if tema:
            ecuaciones_a_probar.extend(cls.ECUACIONES_POR_TEMA.get(tema, []))
        else:
            for ecuaciones_tema in cls.ECUACIONES_POR_TEMA.values():
                ecuaciones_a_probar.extend(ecuaciones_tema)

        # Intentar resolver con cada ecuación
        for ecuacion, var_objetivo_ec in ecuaciones_a_probar:
            if var_objetivo_ec != simbolo_objetivo:
                continue

            try:
                resultado = cls._intentar_resolver(ecuacion, simbolo_objetivo, datos_sym)
                if resultado is not None:
                    return resultado, None
            except Exception:
                continue

        return None, (
            f"No se pudo resolver '{variable_objetivo}' con los datos proporcionados. "
            f"Verifica que tienes suficientes variables conocidas."
        )

    @classmethod
    def _intentar_resolver(
        cls,
        ecuacion: sp.Equality,
        variable: sp.Symbol,
        datos: Dict[sp.Symbol, float]
    ) -> Optional[float]:
        """
        Intenta resolver una ecuación específica con los datos dados.
        """
        # Procesar datos especiales (conversiones de grados a radianes)
        datos_procesados = cls._procesar_datos_especiales(datos)

        # Sustituir valores conocidos
        ecuacion_sustituida = ecuacion.subs(datos_procesados)

        # Verificar si la variable objetivo está en la ecuación
        if variable not in ecuacion_sustituida.free_symbols:
            return None

        # Despejar la variable
        soluciones = sp.solve(ecuacion_sustituida, variable)

        if not soluciones:
            return None

        # Filtrar soluciones reales
        reales = soluciones_reales(soluciones)

        if not reales:
            return None

        # Devolver la primera solución real
        return reales[0]

    @classmethod
    def _procesar_datos_especiales(cls, datos: Dict[sp.Symbol, float]) -> Dict[sp.Symbol, float]:
        """
        Procesa datos que requieren conversiones especiales (ej: grados a radianes).
        """
        datos_procesados = datos.copy()

        # Convertir theta de grados a radianes si está presente
        if Ecuaciones.theta in datos_procesados:
            theta_grados = datos_procesados[Ecuaciones.theta]
            theta_radianes = math.radians(theta_grados)
            datos_procesados[Ecuaciones.theta] = theta_radianes

        return datos_procesados

    @classmethod
    def sugerir_camino(
        cls,
        datos_conocidos: Dict[str, float],
        variable_objetivo: str
    ) -> List[str]:
        """
        Sugiere qué variables adicionales se necesitan para resolver.
        """
        simbolo_objetivo = ParserInput.obtener_simbolo(variable_objetivo)
        if not simbolo_objetivo:
            return [f"Variable '{variable_objetivo}' no reconocida"]

        sugerencias = []

        for tema, ecuaciones in cls.ECUACIONES_POR_TEMA.items():
            for ecuacion, var_objetivo_ec in ecuaciones:
                if var_objetivo_ec == simbolo_objetivo:
                    # Obtener símbolos necesarios para esta ecuación
                    simbolos_necesarios = ecuacion.free_symbols - {simbolo_objetivo}

                    # Verificar cuáles faltan
                    simbolos_conocidos = set()
                    for nombre in datos_conocidos.keys():
                        simbolo = ParserInput.obtener_simbolo(nombre)
                        if simbolo:
                            simbolos_conocidos.add(simbolo)

                    faltantes = simbolos_necesarios - simbolos_conocidos

                    if faltantes:
                        nombres_faltantes = [s.name for s in faltantes]
                        sugerencias.append(
                            f"{tema}: Necesitas {', '.join(nombres_faltantes)}"
                        )

        return sugerencias if sugerencias else ["No se encontró un camino de resolución"]


# ============================================================
# CATÁLOGO DE EJERCICIOS RESOLUBLES
# ============================================================

@dataclasses.dataclass
class TipoEjercicio:
    """
    Describe un tipo de problema que el generador sabe crear.

    Usa DynamicSolver para resolver en lugar de lambdas.
    """

    tema: str
    nombre: str
    variable_objetivo: str  # Variable a calcular (ej: 'x', 'v', 't')
    texto: Callable[[Dict[str, float]], str]
    unidad: str
    rango_respuesta: Tuple[float, float] = (-math.inf, math.inf)

    def resolver(self, datos: Dict[str, float]) -> float:
        """Resuelve el ejercicio usando DynamicSolver."""
        # Pre-procesar datos para conversiones especiales
        datos_procesados = self._preprocesar_datos(datos)

        # Casos especiales: si el valor ya está calculado en el pre-procesamiento
        if self.variable_objetivo == 'v0y' and 'v0y' in datos_procesados:
            return datos_procesados['v0y']
        if self.variable_objetivo == 'v0x' and 'v0x' in datos_procesados:
            return datos_procesados['v0x']
        # Para Circular: T, f, omega calculados directamente
        if self.tema == 'Circular' and self.variable_objetivo in datos_procesados:
            if self.variable_objetivo in ['T', 'f', 'omega']:
                return datos_procesados[self.variable_objetivo]

        resultado, error = DynamicSolver.resolver(
            datos_conocidos=datos_procesados,
            variable_objetivo=self.variable_objetivo,
            tema=self.tema
        )
        if error:
            raise ValueError(f"Error en DynamicSolver: {error}")
        if resultado is None:
            raise ValueError("DynamicSolver devolvió None")
        return resultado

    def _preprocesar_datos(self, datos: Dict[str, float]) -> Dict[str, float]:
        """Pre-procesa datos para conversiones especiales."""
        datos_procesados = datos.copy()

        # Convertir angulo (grados) a theta (radianes)
        if 'angulo' in datos_procesados:
            angulo_grados = datos_procesados['angulo']
            datos_procesados['theta'] = math.radians(angulo_grados)
            # Mantener angulo también por si es necesario para el texto

        # Convertir dx (desplazamiento) a x - x0 para ecuaciones que lo requieren
        if 'dx' in datos_procesados:
            dx = datos_procesados['dx']
            # Si x0 no está presente, asumir 0
            x0 = datos_procesados.get('x0', 0)
            datos_procesados['x'] = x0 + dx
            # Asegurar que x0 esté presente si no lo estaba
            if 'x0' not in datos_procesados:
                datos_procesados['x0'] = 0

        # Calcular componentes de velocidad para movimiento parabólico
        if 'v0' in datos_procesados and 'theta' in datos_procesados:
            v0 = datos_procesados['v0']
            theta = datos_procesados['theta']
            datos_procesados['v0x'] = v0 * math.cos(theta)
            datos_procesados['v0y'] = v0 * math.sin(theta)

        # Para movimiento parabólico, calcular tiempo de vuelo si se necesita para alcance
        if self.variable_objetivo == 'x' and self.tema == 'Parabólico':
            if 'v0y' in datos_procesados and 'g' in datos_procesados:
                datos_procesados['tmax'] = 2 * datos_procesados['v0y'] / datos_procesados['g']
                datos_procesados['t'] = datos_procesados['tmax']  # Usar tmax como tiempo
                if 'x0' not in datos_procesados:
                    datos_procesados['x0'] = 0

        # Para movimiento circular, calcular valores directos cuando sea posible
        if self.tema == 'Circular':
            # Calcular T desde omega directamente
            if self.variable_objetivo == 'T' and 'omega' in datos_procesados:
                datos_procesados['T'] = 2 * math.pi / datos_procesados['omega']
            # Calcular f desde omega directamente
            if self.variable_objetivo == 'f' and 'omega' in datos_procesados:
                datos_procesados['f'] = datos_procesados['omega'] / (2 * math.pi)
            # Calcular omega desde f directamente
            if self.variable_objetivo == 'omega' and 'f' in datos_procesados:
                datos_procesados['omega'] = 2 * math.pi * datos_procesados['f']
            # Mapear v a vc (velocidad tangencial)
            if 'v' in datos_procesados:
                datos_procesados['vc'] = datos_procesados['v']

        # Para "Componente vertical" y "Componente horizontal", si ya tenemos v0x/v0y calculados,
        # devolverlos directamente sin usar DynamicSolver
        if self.variable_objetivo == 'v0y' and 'v0y' in datos_procesados:
            return datos_procesados  # DynamicSolver no se usará, se maneja en resolver
        if self.variable_objetivo == 'v0x' and 'v0x' in datos_procesados:
            return datos_procesados

        return datos_procesados


@dataclasses.dataclass
class Ejercicio:
    tema: str
    nombre: str
    datos: Dict[str, float]
    respuesta: float
    unidad: str
    enunciado: str
    tolerancia: float = 0.01


# ============================================================
# CALCULADORA
# ============================================================

class Calculator:
    """
    Calculadora refactorizada que usa ParserInput y DynamicSolver
    para entrada flexible y resolución dinámica.
    """

    def menu(self):
        while True:
            print("\n" + "=" * 55)
            print("CALCULADORA DE CINEMÁTICA")
            print("=" * 55)
            print("1. Entrada flexible (lenguaje natural)")
            print("2. MRU (modo clásico)")
            print("3. MRUA (modo clásico)")
            print("4. Movimiento parabólico (modo clásico)")
            print("5. Movimiento circular (modo clásico)")
            print("0. Volver")

            opcion = input("\nSelecciona: ").strip()

            if opcion == "1":
                self.entrada_flexible()
            elif opcion == "2":
                self.mru()
            elif opcion == "3":
                self.mrua()
            elif opcion == "4":
                self.parabolico()
            elif opcion == "5":
                self.circular()
            elif opcion == "0":
                return
            else:
                print("Opción inválida.")

    def entrada_flexible(self):
        """Modo de entrada flexible usando ParserInput y DynamicSolver."""
        print("\n--- ENTRADA FLEXIBLE ---")
        print("Ingresa tus datos en formato libre.")
        print("Ejemplos:")
        print("  - calcula x con x0=10, v=5, t=3")
        print("  - v=20, r=2, calcula omega")
        print("  - posicion_inicial=5m, velocidad=10m/s, tiempo=2s")
        print("\nEscribe 'volver' para regresar al menú.\n")

        while True:
            texto = input("Datos: ").strip()

            if texto.lower() in ['volver', 'exit', 'q', 'quit']:
                break

            if not texto:
                continue

            # Parsear entrada
            datos, objetivo = ParserInput.parsear_entrada(texto)

            if not datos:
                print("No se detectaron variables. Intenta con formato: variable=valor")
                continue

            print(f"\nVariables detectadas: {datos}")
            if objetivo:
                print(f"Variable objetivo: {objetivo}")
            else:
                # Si no hay objetivo, pedir al usuario
                print("\nVariables conocidas:")
                for var, val in datos.items():
                    print(f"  {var} = {val}")
                objetivo = input("\n¿Qué variable deseas calcular? ").strip()

            if not objetivo:
                print("Debes especificar una variable para calcular.")
                continue

            # Intentar resolver
            resultado, error = DynamicSolver.resolver(datos, objetivo)

            if error:
                print(f"\n❌ Error: {error}")
                sugerencias = DynamicSolver.sugerir_camino(datos, objetivo)
                print("\nSugerencias:")
                for s in sugerencias:
                    print(f"  - {s}")
            else:
                print(f"\n✅ Resultado: {mostrar_numero(resultado)}")

            input("\nPresiona Enter para continuar...")

    # --------------------------------------------------------
    # Modos clásicos (mantenidos para compatibilidad)
    # --------------------------------------------------------

    def mru(self):
        print("\n--- MRU ---")
        print("1. Calcular posición")
        print("2. Calcular velocidad")
        print("3. Calcular tiempo")
        print("0. Volver")

        opcion = input("Selecciona: ").strip()

        try:
            if opcion == "1":
                x0 = leer_float("x₀ [m]: ")
                v = leer_float("v [m/s]: ")
                t = leer_float("t [s]: ", permitir_negativo=False)

                resultado, _ = DynamicSolver.resolver(
                    {'x0': x0, 'v': v, 't': t},
                    'x',
                    'MRU'
                )
                print(f"\nPosición: {mostrar_numero(resultado)} m")

            elif opcion == "2":
                x0 = leer_float("x₀ [m]: ")
                x = leer_float("x [m]: ")
                t = leer_float(
                    "t [s]: ",
                    permitir_negativo=False,
                    permitir_cero=False
                )

                resultado, _ = DynamicSolver.resolver(
                    {'x0': x0, 'x': x, 't': t},
                    'v',
                    'MRU'
                )
                print(f"\nVelocidad: {mostrar_numero(resultado)} m/s")

            elif opcion == "3":
                x0 = leer_float("x₀ [m]: ")
                x = leer_float("x [m]: ")
                v = leer_float(
                    "v [m/s]: ",
                    permitir_cero=False
                )

                resultado, _ = DynamicSolver.resolver(
                    {'x0': x0, 'x': x, 'v': v},
                    't',
                    'MRU'
                )
                print(f"\nTiempo: {mostrar_numero(resultado)} s")

        except ValueError as error:
            print(f"\nError: {error}")

    # --------------------------------------------------------
    # MRUA
    # --------------------------------------------------------

    def mrua(self):
        print("\n--- MRUA ---")
        print("1. Velocidad final")
        print("2. Desplazamiento")
        print("3. Aceleración")
        print("4. Velocidad media")
        print("5. Velocidad final sin tiempo")
        print("0. Volver")

        opcion = input("Selecciona: ").strip()

        try:
            if opcion == "1":
                v0 = leer_float("v₀ [m/s]: ")
                a = leer_float("a [m/s²]: ")
                t = leer_float("t [s]: ", permitir_negativo=False)

                resultado, _ = DynamicSolver.resolver(
                    {'v0': v0, 'a': a, 't': t},
                    'vf',
                    'MRUA'
                )
                print(f"\nVelocidad final: {mostrar_numero(resultado)} m/s")

            elif opcion == "2":
                x0 = leer_float("x₀ [m]: ")
                v0 = leer_float("v₀ [m/s]: ")
                a = leer_float("a [m/s²]: ")
                t = leer_float("t [s]: ", permitir_negativo=False)

                resultado, _ = DynamicSolver.resolver(
                    {'x0': x0, 'v0': v0, 'a': a, 't': t},
                    'x',
                    'MRUA'
                )
                print(f"\nPosición final: {mostrar_numero(resultado)} m")
                print(
                    f"Desplazamiento: "
                    f"{mostrar_numero(resultado - x0)} m"
                )

            elif opcion == "3":
                v0 = leer_float("v₀ [m/s]: ")
                vf = leer_float("vf [m/s]: ")
                t = leer_float(
                    "t [s]: ",
                    permitir_negativo=False,
                    permitir_cero=False
                )

                resultado, _ = DynamicSolver.resolver(
                    {'v0': v0, 'vf': vf, 't': t},
                    'a',
                    'MRUA'
                )
                print(f"\nAceleración: {mostrar_numero(resultado)} m/s²")

            elif opcion == "4":
                v0 = leer_float("v₀ [m/s]: ")
                vf = leer_float("vf [m/s]: ")

                resultado, _ = DynamicSolver.resolver(
                    {'v0': v0, 'vf': vf},
                    'vmedia',
                    'MRUA'
                )
                print(f"\nVelocidad media: {mostrar_numero(resultado)} m/s")

            elif opcion == "5":
                v0 = leer_float("v₀ [m/s]: ")
                a = leer_float("a [m/s²]: ")
                dx = leer_float("Δx [m]: ")

                resultado, _ = DynamicSolver.resolver(
                    {'v0': v0, 'a': a, 'dx': dx},
                    'vf',
                    'MRUA'
                )
                print(f"\nVelocidad final: {mostrar_numero(resultado)} m/s")

        except ValueError as error:
            print(f"\nError: {error}")

    # --------------------------------------------------------
    # PARABÓLICO
    # --------------------------------------------------------

    def parabolico(self):
        print("\n--- MOVIMIENTO PARABÓLICO ---")
        print("1. Componentes de velocidad")
        print("2. Altura máxima")
        print("3. Tiempo de vuelo (misma altura)")
        print("4. Alcance (misma altura)")
        print("0. Volver")

        opcion = input("Selecciona: ").strip()

        try:
            if opcion == "1":
                v0 = leer_float(
                    "v₀ [m/s]: ",
                    permitir_negativo=False
                )
                angulo = leer_float(
                    "Ángulo [grados]: ",
                    permitir_negativo=False
                )

                theta_rad = grados_a_radianes(angulo)

                vx = v0 * math.cos(theta_rad)
                vy = v0 * math.sin(theta_rad)

                print(f"\nv₀x = {mostrar_numero(vx)} m/s")
                print(f"v₀y = {mostrar_numero(vy)} m/s")

            elif opcion == "2":
                v0 = leer_float("v₀ [m/s]: ", permitir_negativo=False)
                angulo = leer_float(
                    "Ángulo [grados]: ",
                    permitir_negativo=False
                )
                y0 = leer_float("Altura inicial y₀ [m]: ", permitir_negativo=False)

                resultado, _ = DynamicSolver.resolver(
                    {'v0': v0, 'angulo': angulo, 'y0': y0, 'g': G},
                    'hmax',
                    'Parabólico'
                )
                print(f"\nAltura máxima: {mostrar_numero(resultado)} m")

            elif opcion == "3":
                v0 = leer_float("v₀ [m/s]: ", permitir_negativo=False)
                angulo = leer_float(
                    "Ángulo [grados]: ",
                    permitir_negativo=False
                )

                resultado, _ = DynamicSolver.resolver(
                    {'v0': v0, 'angulo': angulo, 'g': G},
                    'tmax',
                    'Parabólico'
                )
                print(f"\nTiempo de vuelo: {mostrar_numero(resultado)} s")

            elif opcion == "4":
                v0 = leer_float("v₀ [m/s]: ", permitir_negativo=False)
                angulo = leer_float(
                    "Ángulo [grados]: ",
                    permitir_negativo=False
                )

                resultado, _ = DynamicSolver.resolver(
                    {'v0': v0, 'angulo': angulo, 'g': G},
                    'x',
                    'Parabólico'
                )
                print(f"\nAlcance: {mostrar_numero(resultado)} m")

        except ValueError as error:
            print(f"\nError: {error}")

    # --------------------------------------------------------
    # CIRCULAR
    # --------------------------------------------------------

    def circular(self):
        print("\n--- MOVIMIENTO CIRCULAR ---")
        print("1. Velocidad tangencial")
        print("2. Velocidad angular")
        print("3. Período")
        print("4. Frecuencia")
        print("5. Aceleración centrípeta")
        print("0. Volver")

        opcion = input("Selecciona: ").strip()

        try:
            if opcion == "1":
                omega = leer_float(
                    "ω [rad/s]: ",
                    permitir_negativo=False
                )
                r = leer_float(
                    "Radio [m]: ",
                    permitir_negativo=False
                )

                resultado, _ = DynamicSolver.resolver(
                    {'omega': omega, 'r': r},
                    'vc',
                    'Circular'
                )
                print(f"\nv = {mostrar_numero(resultado)} m/s")

            elif opcion == "2":
                f = leer_float(
                    "Frecuencia [Hz]: ",
                    permitir_negativo=False
                )

                resultado, _ = DynamicSolver.resolver(
                    {'f': f},
                    'omega',
                    'Circular'
                )
                print(f"\nω = {mostrar_numero(resultado)} rad/s")

            elif opcion == "3":
                omega = leer_float(
                    "ω [rad/s]: ",
                    permitir_negativo=False,
                    permitir_cero=False
                )

                resultado, _ = DynamicSolver.resolver(
                    {'omega': omega},
                    'T',
                    'Circular'
                )
                print(f"\nT = {mostrar_numero(resultado)} s")

            elif opcion == "4":
                T = leer_float(
                    "T [s]: ",
                    permitir_negativo=False,
                    permitir_cero=False
                )

                resultado, _ = DynamicSolver.resolver(
                    {'T': T},
                    'f',
                    'Circular'
                )
                print(f"\nf = {mostrar_numero(resultado)} Hz")

            elif opcion == "5":
                print("1. Usando velocidad tangencial")
                print("2. Usando velocidad angular")

                subopcion = input("Selecciona: ").strip()

                r = leer_float(
                    "Radio [m]: ",
                    permitir_negativo=False,
                    permitir_cero=False
                )

                if subopcion == "1":
                    v = leer_float(
                        "v [m/s]: ",
                        permitir_negativo=False
                    )
                    resultado, _ = DynamicSolver.resolver(
                        {'v': v, 'r': r},
                        'ac',
                        'Circular'
                    )

                elif subopcion == "2":
                    omega = leer_float(
                        "ω [rad/s]: ",
                        permitir_negativo=False
                    )
                    resultado, _ = DynamicSolver.resolver(
                        {'omega': omega, 'r': r},
                        'ac',
                        'Circular'
                    )

                else:
                    print("Opción inválida.")
                    return

                print(
                    f"\naᶜ = {mostrar_numero(resultado)} m/s²"
                )

        except ValueError as error:
            print(f"\nError: {error}")


# ============================================================
# SIMULADOR
# ============================================================

class Simulator:

    def menu(self):
        while True:
            print("\n" + "=" * 55)
            print("SIMULACIÓN Y GRÁFICAS")
            print("=" * 55)
            print("1. MRU")
            print("2. MRUA")
            print("3. Movimiento parabólico")
            print("4. Movimiento circular")
            print("0. Volver")

            opcion = input("\nSelecciona: ").strip()

            if opcion == "1":
                self.mru()
            elif opcion == "2":
                self.mrua()
            elif opcion == "3":
                self.parabolico()
            elif opcion == "4":
                self.circular()
            elif opcion == "0":
                return
            else:
                print("Opción inválida.")

    def mru(self):
        print("\n--- SIMULACIÓN MRU ---")

        x0 = leer_float("Posición inicial x₀ [m]: ")
        v = leer_float("Velocidad [m/s]: ")
        duracion = leer_float(
            "Duración [s]: ",
            permitir_negativo=False
        )

        t = np.linspace(0, duracion, 500)
        x = x0 + v * t

        fig, ax = plt.subplots(figsize=(9, 5))
        line, = ax.plot([], [], linewidth=2, color='blue')
        point, = ax.plot([], [], 'ro', markersize=8)

        ax.set_xlim(0, duracion)
        ax.set_ylim(min(x) - 1, max(x) + 1)
        ax.set_title("MRU - Posición vs Tiempo")
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Posición [m]")
        ax.grid(True)

        def init():
            line.set_data([], [])
            point.set_data([], [])
            return line, point

        def animate(i):
            line.set_data(t[:i], x[:i])
            point.set_data([t[i]], [x[i]])
            return line, point

        ani = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=len(t), interval=20, blit=True, repeat=False
        )

        plt.tight_layout()
        plt.show()

    def mrua(self):
        print("\n--- SIMULACIÓN MRUA ---")

        x0 = leer_float("Posición inicial x₀ [m]: ")
        v0 = leer_float("Velocidad inicial v₀ [m/s]: ")
        a = leer_float("Aceleración [m/s²]: ")
        duracion = leer_float(
            "Duración [s]: ",
            permitir_negativo=False
        )

        t = np.linspace(0, duracion, 500)

        x = x0 + v0 * t + 0.5 * a * t**2
        v = v0 + a * t

        fig, axes = plt.subplots(2, 1, figsize=(9, 8))

        line_x, = axes[0].plot([], [], linewidth=2, color='blue')
        point_x, = axes[0].plot([], [], 'ro', markersize=8)

        axes[0].set_xlim(0, duracion)
        axes[0].set_ylim(min(x) - 1, max(x) + 1)
        axes[0].set_title("MRUA - Posición vs Tiempo")
        axes[0].set_xlabel("Tiempo [s]")
        axes[0].set_ylabel("Posición [m]")
        axes[0].grid(True)

        line_v, = axes[1].plot([], [], linewidth=2, color='green')
        point_v, = axes[1].plot([], [], 'ro', markersize=8)

        axes[1].set_xlim(0, duracion)
        axes[1].set_ylim(min(v) - 1, max(v) + 1)
        axes[1].set_title("MRUA - Velocidad vs Tiempo")
        axes[1].set_xlabel("Tiempo [s]")
        axes[1].set_ylabel("Velocidad [m/s]")
        axes[1].grid(True)

        def init():
            line_x.set_data([], [])
            point_x.set_data([], [])
            line_v.set_data([], [])
            point_v.set_data([], [])
            return line_x, point_x, line_v, point_v

        def animate(i):
            line_x.set_data(t[:i], x[:i])
            point_x.set_data([t[i]], [x[i]])
            line_v.set_data(t[:i], v[:i])
            point_v.set_data([t[i]], [v[i]])
            return line_x, point_x, line_v, point_v

        ani = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=len(t), interval=20, blit=True, repeat=False
        )

        plt.tight_layout()
        plt.show()

    def parabolico(self):
        print("\n--- SIMULACIÓN MOVIMIENTO PARABÓLICO ---")

        v0 = leer_float(
            "Velocidad inicial v₀ [m/s]: ",
            permitir_negativo=False
        )
        angulo = leer_float(
            "Ángulo [grados]: ",
            permitir_negativo=False
        )
        y0 = leer_float(
            "Altura inicial y₀ [m]: ",
            permitir_negativo=False
        )

        theta = math.radians(angulo)

        vx0 = v0 * math.cos(theta)
        vy0 = v0 * math.sin(theta)

        # Solución positiva del tiempo cuando y = 0.
        # Si y0 = 0, se obtiene t = 0 y el tiempo de vuelo.
        discriminante = vy0**2 + 2 * G * y0
        tiempo_vuelo = (
            vy0 + math.sqrt(discriminante)
        ) / G

        t = np.linspace(0, tiempo_vuelo, 600)

        x = vx0 * t
        y = y0 + vy0 * t - 0.5 * G * t**2

        vx = np.full_like(t, vx0)
        vy = vy0 - G * t
        velocidad = np.sqrt(vx**2 + vy**2)

        hmax = y0 + vy0**2 / (2 * G)
        alcance = x[-1]

        # Gráfica animada de trayectoria
        fig, ax = plt.subplots(figsize=(10, 6))

        line, = ax.plot([], [], linewidth=2, color='blue')
        point, = ax.plot([], [], 'ro', markersize=8)

        ax.set_xlim(-1, alcance * 1.1)
        ax.set_ylim(-1, hmax * 1.2)
        ax.set_title("Movimiento Parabólico")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True)
        ax.axhline(0, linewidth=1)

        texto = (
            f"Altura máxima = {hmax:.2f} m\n"
            f"Alcance = {alcance:.2f} m\n"
            f"Tiempo de vuelo = {tiempo_vuelo:.2f} s"
        )

        ax.text(
            0.02,
            0.97,
            texto,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", alpha=0.8)
        )

        def init_traj():
            line.set_data([], [])
            point.set_data([], [])
            return line, point

        def animate_traj(i):
            line.set_data(x[:i], y[:i])
            point.set_data([x[i]], [y[i]])
            return line, point

        ani_traj = animation.FuncAnimation(
            fig, animate_traj, init_func=init_traj,
            frames=len(t), interval=20, blit=True, repeat=False
        )

        plt.tight_layout()
        plt.show()

        # Gráfica animada de velocidades
        fig, ax = plt.subplots(figsize=(9, 5))

        line_vx, = ax.plot([], [], label="vx", color='blue')
        line_vy, = ax.plot([], [], label="vy", color='green')
        line_v, = ax.plot([], [], label="|v|", color='red')
        point_vx, = ax.plot([], [], 'bo', markersize=6)
        point_vy, = ax.plot([], [], 'go', markersize=6)
        point_v, = ax.plot([], [], 'ro', markersize=6)

        ax.set_xlim(0, tiempo_vuelo)
        ax.set_ylim(min(velocidad) - 1, max(velocidad) + 1)
        ax.set_title("Movimiento Parabólico - Velocidades")
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Velocidad [m/s]")
        ax.legend()
        ax.grid(True)

        def init_vel():
            line_vx.set_data([], [])
            line_vy.set_data([], [])
            line_v.set_data([], [])
            point_vx.set_data([], [])
            point_vy.set_data([], [])
            point_v.set_data([], [])
            return line_vx, line_vy, line_v, point_vx, point_vy, point_v

        def animate_vel(i):
            line_vx.set_data(t[:i], vx[:i])
            line_vy.set_data(t[:i], vy[:i])
            line_v.set_data(t[:i], velocidad[:i])
            point_vx.set_data([t[i]], [vx[i]])
            point_vy.set_data([t[i]], [vy[i]])
            point_v.set_data([t[i]], [velocidad[i]])
            return line_vx, line_vy, line_v, point_vx, point_vy, point_v

        ani_vel = animation.FuncAnimation(
            fig, animate_vel, init_func=init_vel,
            frames=len(t), interval=20, blit=True, repeat=False
        )

        plt.tight_layout()
        plt.show()

    def circular(self):
        print("\n--- SIMULACIÓN MOVIMIENTO CIRCULAR ---")

        r = leer_float(
            "Radio [m]: ",
            permitir_negativo=False,
            permitir_cero=False
        )
        omega = leer_float(
            "Velocidad angular ω [rad/s]: ",
            permitir_negativo=False
        )
        duracion = leer_float(
            "Duración [s]: ",
            permitir_negativo=False
        )

        t = np.linspace(0, duracion, 800)
        theta = omega * t

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        velocidad = omega * r
        aceleracion = omega**2 * r

        fig, ax = plt.subplots(figsize=(7, 7))

        line, = ax.plot([], [], linestyle="--", color='blue')
        point, = ax.plot([], [], 'ro', markersize=8)
        center, = ax.plot([0], [0], 'ko', markersize=10, label="Centro")

        # Quivers para vectores (inicialmente vacíos)
        quiver_v = ax.quiver([], [], [], [], angles="xy", scale_units="xy", scale=1, width=0.004, color='green', label="Velocidad tangencial")
        quiver_a = ax.quiver([], [], [], [], angles="xy", scale_units="xy", scale=1, width=0.004, color='red', label="Aceleración centrípeta")

        limite = r * 1.4
        ax.set_xlim(-limite, limite)
        ax.set_ylim(-limite, limite)

        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Movimiento Circular")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True)
        ax.legend()

        texto = (
            f"v = {velocidad:.2f} m/s\n"
            f"aᶜ = {aceleracion:.2f} m/s²\n"
            f"ω = {omega:.2f} rad/s"
        )

        ax.text(
            0.02,
            0.97,
            texto,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", alpha=0.8)
        )

        def init():
            line.set_data([], [])
            point.set_data([], [])
            quiver_v.set_UVC([], [])
            quiver_v.set_offsets(np.array([[], []]).T)
            quiver_a.set_UVC([], [])
            quiver_a.set_offsets(np.array([[], []]).T)
            return line, point, quiver_v, quiver_a

        def animate(i):
            line.set_data(x[:i], y[:i])
            point.set_data([x[i]], [y[i]])

            # Calcular vectores en el punto actual
            vx = -velocidad * math.sin(theta[i])
            vy = velocidad * math.cos(theta[i])
            ax_val = -x[i] * omega**2
            ay_val = -y[i] * omega**2

            # Actualizar quivers
            quiver_v.set_offsets(np.array([[x[i]], [y[i]]]).T)
            quiver_v.set_UVC([vx], [vy])

            quiver_a.set_offsets(np.array([[x[i]], [y[i]]]).T)
            quiver_a.set_UVC([ax_val], [ay_val])

            return line, point, quiver_v, quiver_a

        ani = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=len(t), interval=20, blit=False, repeat=False
        )

        plt.tight_layout()
        plt.show()


# ============================================================
# GENERADOR DE DESAFÍOS
# ============================================================

@dataclasses.dataclass
class RegistroEjercicio:
    tema: str
    nombre: str
    correcto: bool
    tiempo_segundos: float
    fecha: str


class GestorEstadisticas:
    """Gestiona el almacenamiento y análisis de estadísticas de ejercicios."""

    def __init__(self, archivo: str = "estadisticas_cinematica.json"):
        self.archivo = Path(archivo)
        self.datos = self._cargar()

    def _cargar(self) -> dict:
        """Carga los datos desde el archivo JSON."""
        if self.archivo.exists():
            with open(self.archivo, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"registros": [], "resumen": {}}

    def _guardar(self):
        """Guarda los datos en el archivo JSON."""
        with open(self.archivo, 'w', encoding='utf-8') as f:
            json.dump(self.datos, f, indent=2, ensure_ascii=False)

    def registrar(self, registro: RegistroEjercicio):
        """Registra un ejercicio completado y actualiza el resumen."""
        self.datos["registros"].append(dataclasses.asdict(registro))
        self._actualizar_resumen()
        self._guardar()

    def _actualizar_resumen(self):
        """Actualiza el resumen de estadísticas por tipo de ejercicio."""
        registros = self.datos["registros"]
        resumen = {}

        for reg in registros:
            tema = reg["tema"]
            nombre = reg["nombre"]
            clave = f"{tema}|{nombre}"

            if clave not in resumen:
                resumen[clave] = {
                    "tema": tema,
                    "nombre": nombre,
                    "total": 0,
                    "correctas": 0,
                    "tiempo_total": 0.0,
                    "tiempos": []
                }

            resumen[clave]["total"] += 1
            if reg["correcto"]:
                resumen[clave]["correctas"] += 1
            resumen[clave]["tiempo_total"] += reg["tiempo_segundos"]
            resumen[clave]["tiempos"].append(reg["tiempo_segundos"])

        # Calcular promedios y mediana
        for clave, datos in resumen.items():
            if datos["total"] > 0:
                datos["porcentaje"] = (datos["correctas"] / datos["total"]) * 100
                datos["tiempo_promedio"] = datos["tiempo_total"] / datos["total"]
                datos["tiempos"].sort()
                n = len(datos["tiempos"])
                datos["tiempo_mediana"] = datos["tiempos"][n // 2] if n > 0 else 0

        self.datos["resumen"] = resumen

    def obtener_estadisticas(self) -> dict:
        """Devuelve todas las estadísticas almacenadas."""
        return self.datos


class Challenge:

    def __init__(self):
        self.tipos = self._crear_catalogo()

    def _crear_catalogo(self) -> List[TipoEjercicio]:
        return [
            # ==================================================
            # MRU
            # ==================================================

            TipoEjercicio(
                tema="MRU",
                nombre="Calcular posición",
                variable_objetivo="x",
                texto=lambda d: (
                    f"Un objeto parte desde x₀ = {d['x0']:.2f} m "
                    f"y se mueve con velocidad constante de "
                    f"{d['v']:.2f} m/s durante {d['t']:.2f} s.\n"
                    f"¿Cuál es su posición final?"
                ),
                unidad="m"
            ),

            TipoEjercicio(
                tema="MRU",
                nombre="Calcular velocidad",
                variable_objetivo="v",
                texto=lambda d: (
                    f"Un objeto pasa de x₀ = {d['x0']:.2f} m "
                    f"a x = {d['x']:.2f} m en {d['t']:.2f} s.\n"
                    f"¿Cuál es su velocidad media?"
                ),
                unidad="m/s"
            ),

            TipoEjercicio(
                tema="MRU",
                nombre="Calcular tiempo",
                variable_objetivo="t",
                texto=lambda d: (
                    f"Un objeto parte de x₀ = {d['x0']:.2f} m "
                    f"y se mueve a {d['v']:.2f} m/s hasta llegar "
                    f"a x = {d['x']:.2f} m.\n"
                    f"¿Cuánto tiempo tarda?"
                ),
                unidad="s"
            ),

            # ==================================================
            # MRUA
            # ==================================================

            TipoEjercicio(
                tema="MRUA",
                nombre="Velocidad final",
                variable_objetivo="vf",
                texto=lambda d: (
                    f"Un automóvil tiene una velocidad inicial de "
                    f"{d['v0']:.2f} m/s y acelera a "
                    f"{d['a']:.2f} m/s² durante {d['t']:.2f} s.\n"
                    f"¿Cuál es su velocidad final?"
                ),
                unidad="m/s"
            ),

            TipoEjercicio(
                tema="MRUA",
                nombre="Desplazamiento",
                variable_objetivo="x",
                texto=lambda d: (
                    f"Un objeto parte con velocidad inicial "
                    f"{d['v0']:.2f} m/s y aceleración constante de "
                    f"{d['a']:.2f} m/s² durante {d['t']:.2f} s.\n"
                    f"¿Qué desplazamiento realiza?"
                ),
                unidad="m"
            ),

            TipoEjercicio(
                tema="MRUA",
                nombre="Aceleración",
                variable_objetivo="a",
                texto=lambda d: (
                    f"Un vehículo cambia su velocidad de "
                    f"{d['v0']:.2f} m/s a {d['vf']:.2f} m/s "
                    f"en {d['t']:.2f} s.\n"
                    f"¿Cuál es su aceleración?"
                ),
                unidad="m/s²"
            ),

            TipoEjercicio(
                tema="MRUA",
                nombre="Velocidad media",
                variable_objetivo="vmedia",
                texto=lambda d: (
                    f"Un móvil pasa de {d['v0']:.2f} m/s a "
                    f"{d['vf']:.2f} m/s con aceleración constante.\n"
                    f"¿Cuál es su velocidad media?"
                ),
                unidad="m/s"
            ),

            TipoEjercicio(
                tema="MRUA",
                nombre="Velocidad sin tiempo",
                variable_objetivo="vf",
                texto=lambda d: (
                    f"Un móvil parte con velocidad {d['v0']:.2f} m/s, "
                    f"tiene una aceleración de {d['a']:.2f} m/s² "
                    f"y recorre {d['dx']:.2f} m.\n"
                    f"¿Cuál es su velocidad final?"
                ),
                unidad="m/s"
            ),

            # ==================================================
            # PARABÓLICO
            # ==================================================

            TipoEjercicio(
                tema="Parabólico",
                nombre="Componente horizontal",
                variable_objetivo="v0x",
                texto=lambda d: (
                    f"Un proyectil es lanzado con velocidad inicial "
                    f"{d['v0']:.2f} m/s formando un ángulo de "
                    f"{d['angulo']:.2f}°.\n"
                    f"¿Cuál es la componente horizontal de su velocidad?"
                ),
                unidad="m/s"
            ),

            TipoEjercicio(
                tema="Parabólico",
                nombre="Componente vertical",
                variable_objetivo="v0y",
                texto=lambda d: (
                    f"Un proyectil es lanzado con velocidad inicial "
                    f"{d['v0']:.2f} m/s formando un ángulo de "
                    f"{d['angulo']:.2f}°.\n"
                    f"¿Cuál es la componente vertical de su velocidad?"
                ),
                unidad="m/s"
            ),

            TipoEjercicio(
                tema="Parabólico",
                nombre="Altura máxima",
                variable_objetivo="hmax",
                texto=lambda d: (
                    f"Un proyectil se lanza desde el suelo con "
                    f"una velocidad inicial de {d['v0']:.2f} m/s "
                    f"y un ángulo de {d['angulo']:.2f}°.\n"
                    f"¿Cuál es su altura máxima?"
                ),
                unidad="m"
            ),

            TipoEjercicio(
                tema="Parabólico",
                nombre="Tiempo de vuelo",
                variable_objetivo="tmax",
                texto=lambda d: (
                    f"Un proyectil se lanza desde el suelo con "
                    f"{d['v0']:.2f} m/s a {d['angulo']:.2f}°.\n"
                    f"¿Cuál es su tiempo total de vuelo?"
                ),
                unidad="s"
            ),

            TipoEjercicio(
                tema="Parabólico",
                nombre="Alcance",
                variable_objetivo="x",
                texto=lambda d: (
                    f"Un proyectil se lanza desde el suelo con "
                    f"{d['v0']:.2f} m/s a {d['angulo']:.2f}°.\n"
                    f"¿Cuál es su alcance horizontal?"
                ),
                unidad="m"
            ),

            # ==================================================
            # CIRCULAR
            # ==================================================

            TipoEjercicio(
                tema="Circular",
                nombre="Velocidad tangencial",
                variable_objetivo="vc",
                texto=lambda d: (
                    f"Una partícula gira con radio {d['r']:.2f} m "
                    f"y velocidad angular {d['omega']:.2f} rad/s.\n"
                    f"¿Cuál es su velocidad tangencial?"
                ),
                unidad="m/s"
            ),

            TipoEjercicio(
                tema="Circular",
                nombre="Período",
                variable_objetivo="T",
                texto=lambda d: (
                    f"Una partícula gira con velocidad angular "
                    f"{d['omega']:.2f} rad/s.\n"
                    f"¿Cuál es su período?"
                ),
                unidad="s"
            ),

            TipoEjercicio(
                tema="Circular",
                nombre="Frecuencia",
                variable_objetivo="f",
                texto=lambda d: (
                    f"Una partícula gira con velocidad angular "
                    f"{d['omega']:.2f} rad/s.\n"
                    f"¿Cuál es su frecuencia?"
                ),
                unidad="Hz"
            ),

            TipoEjercicio(
                tema="Circular",
                nombre="Velocidad angular",
                variable_objetivo="omega",
                texto=lambda d: (
                    f"Una partícula gira con una frecuencia de "
                    f"{d['f']:.2f} Hz.\n"
                    f"¿Cuál es su velocidad angular?"
                ),
                unidad="rad/s"
            ),

            TipoEjercicio(
                tema="Circular",
                nombre="Aceleración centrípeta",
                variable_objetivo="ac",
                texto=lambda d: (
                    f"Una partícula se mueve en un círculo de "
                    f"radio {d['r']:.2f} m con velocidad "
                    f"{d['v']:.2f} m/s.\n"
                    f"¿Cuál es su aceleración centrípeta?"
                ),
                unidad="m/s²"
            ),
        ]

    # --------------------------------------------------------
    # Generación de datos
    # --------------------------------------------------------

    def _datos_para(self, ejercicio: TipoEjercicio) -> Dict[str, float]:
        """
        Genera datos válidos específicamente para cada tipo.
        Esto evita crear problemas sin solución física.
        """

        nombre = ejercicio.nombre

        if ejercicio.tema == "MRU":

            if nombre == "Calcular posición":
                return {
                    "x0": random.uniform(0, 50),
                    "v": random.uniform(2, 30),
                    "t": random.uniform(1, 20)
                }

            if nombre == "Calcular velocidad":
                x0 = random.uniform(0, 20)
                v = random.uniform(2, 30)
                t = random.uniform(1, 20)

                return {
                    "x0": x0,
                    "x": x0 + v * t,
                    "t": t
                }

            if nombre == "Calcular tiempo":
                x0 = random.uniform(0, 20)
                v = random.uniform(2, 30)
                t = random.uniform(1, 20)

                return {
                    "x0": x0,
                    "x": x0 + v * t,
                    "v": v
                }

        if ejercicio.tema == "MRUA":

            if nombre == "Velocidad final":
                return {
                    "v0": random.uniform(0, 20),
                    "a": random.uniform(1, 8),
                    "t": random.uniform(1, 10)
                }

            if nombre == "Desplazamiento":
                return {
                    "x0": random.uniform(0, 20),
                    "v0": random.uniform(0, 20),
                    "a": random.uniform(1, 8),
                    "t": random.uniform(1, 10)
                }

            if nombre == "Aceleración":
                v0 = random.uniform(0, 15)
                a = random.uniform(1, 8)
                t = random.uniform(1, 10)

                return {
                    "v0": v0,
                    "vf": v0 + a * t,
                    "t": t
                }

            if nombre == "Velocidad media":
                v0 = random.uniform(0, 20)
                vf = random.uniform(v0 + 2, 40)

                return {
                    "v0": v0,
                    "vf": vf
                }

            if nombre == "Velocidad sin tiempo":
                v0 = random.uniform(0, 15)
                a = random.uniform(1, 8)
                dx = random.uniform(2, 50)

                return {
                    "v0": v0,
                    "a": a,
                    "dx": dx
                }

        if ejercicio.tema == "Parabólico":

            v0 = random.uniform(10, 40)
            angulo = random.uniform(20, 70)

            if nombre in (
                "Componente horizontal",
                "Componente vertical"
            ):
                return {
                    "v0": v0,
                    "angulo": angulo
                }

            if nombre == "Altura máxima":
                return {
                    "v0": v0,
                    "angulo": angulo,
                    "y0": 0,
                    "g": G
                }

            if nombre == "Tiempo de vuelo":
                return {
                    "v0": v0,
                    "angulo": angulo,
                    "g": G
                }

            if nombre == "Alcance":
                return {
                    "v0": v0,
                    "angulo": angulo,
                    "g": G
                }

        if ejercicio.tema == "Circular":

            if nombre == "Velocidad tangencial":
                return {
                    "r": random.uniform(0.5, 10),
                    "omega": random.uniform(1, 20)
                }

            if nombre == "Período":
                return {
                    "omega": random.uniform(1, 20)
                }

            if nombre == "Frecuencia":
                return {
                    "omega": random.uniform(1, 20)
                }

            if nombre == "Velocidad angular":
                return {
                    "f": random.uniform(0.5, 10)
                }

            if nombre == "Aceleración centrípeta":
                return {
                    "r": random.uniform(0.5, 10),
                    "v": random.uniform(1, 30)
                }

        raise ValueError(
            f"No existe un generador de datos para: "
            f"{ejercicio.tema} - {ejercicio.nombre}"
        )

    # --------------------------------------------------------
    # Crear ejercicio
    # --------------------------------------------------------

    def generar(self, tema: Optional[str] = None) -> Ejercicio:

        candidatos = self.tipos

        if tema:
            candidatos = [
                e for e in candidatos
                if e.tema.lower() == tema.lower()
            ]

        if not candidatos:
            raise ValueError("No hay ejercicios disponibles para ese tema.")

        tipo = random.choice(candidatos)

        # Reintenta por seguridad si alguna combinación produce
        # una respuesta inválida.
        for _ in range(50):

            datos = self._datos_para(tipo)

            try:
                respuesta = float(tipo.resolver(datos))

                if math.isfinite(respuesta):
                    enunciado = tipo.texto(datos)

                    # Tolerancia adaptativa:
                    # suficiente para ejercicios con decimales.
                    tolerancia = max(
                        0.01,
                        abs(respuesta) * 0.005
                    )

                    return Ejercicio(
                        tema=tipo.tema,
                        nombre=tipo.nombre,
                        datos=datos,
                        respuesta=respuesta,
                        unidad=tipo.unidad,
                        enunciado=enunciado,
                        tolerancia=tolerancia
                    )

            except (ValueError, ZeroDivisionError, OverflowError):
                pass

        raise RuntimeError(
            f"No fue posible generar un ejercicio válido para {tipo.tema}/{tipo.nombre}."
        )

    # --------------------------------------------------------
    # Verificación
    # --------------------------------------------------------

    def verificar(
        self,
        ejercicio: Ejercicio,
        respuesta_usuario: float
    ) -> bool:

        return abs(
            respuesta_usuario - ejercicio.respuesta
        ) <= ejercicio.tolerancia

    # --------------------------------------------------------
    # Menú
    # --------------------------------------------------------

    def menu(self):

        while True:

            print("\n" + "=" * 55)
            print("MODO DESAFÍO")
            print("=" * 55)
            print("1. Tema aleatorio")
            print("2. Solo MRU")
            print("3. Solo MRUA")
            print("4. Solo movimiento parabólico")
            print("5. Solo movimiento circular")
            print("0. Volver")

            opcion = input("\nSelecciona: ").strip()

            if opcion == "0":
                return

            temas = {
                "1": None,
                "2": "MRU",
                "3": "MRUA",
                "4": "Parabólico",
                "5": "Circular"
            }

            if opcion not in temas:
                print("Opción inválida.")
                continue

            self.jugar(temas[opcion])

    def jugar(self, tema: Optional[str]):
        gestor = GestorEstadisticas()
        puntuacion = 0
        total = 0

        while True:
            ejercicio = self.generar(tema)

            # Iniciar temporizador
            tiempo_inicio = time.time()

            print("\n" + "=" * 65)
            print(f"TEMA: {ejercicio.tema}")
            print(f"TIPO: {ejercicio.nombre}")
            print("=" * 65)
            print(ejercicio.enunciado)
            print()

            respuesta_usuario_texto = input(
                f"Respuesta [{ejercicio.unidad}] "
                "(q para salir): "
            ).strip()

            if respuesta_usuario_texto.lower() == "q":
                break

            try:
                respuesta_usuario = float(respuesta_usuario_texto)
            except ValueError:
                print("Debes ingresar un número.")
                continue

            # Calcular tiempo transcurrido
            tiempo_fin = time.time()
            tiempo_empleado = tiempo_fin - tiempo_inicio

            total += 1
            correcto = self.verificar(ejercicio, respuesta_usuario)

            if correcto:
                puntuacion += 1
                print("\n✓ ¡CORRECTO!")
            else:
                print("\n✗ INCORRECTO")
                print(
                    "Respuesta correcta: "
                    f"{mostrar_numero(ejercicio.respuesta)} "
                    f"{ejercicio.unidad}"
                )

            # Registrar en estadísticas
            registro = RegistroEjercicio(
                tema=ejercicio.tema,
                nombre=ejercicio.nombre,
                correcto=correcto,
                tiempo_segundos=tiempo_empleado,
                fecha=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            gestor.registrar(registro)

            porcentaje = (
                puntuacion / total * 100
                if total
                else 0
            )

            print(
                f"Puntuación: {puntuacion}/{total} "
                f"({porcentaje:.1f}%)"
            )
            print(f"Tiempo: {tiempo_empleado:.2f} segundos")

            input("\nPresiona Enter para continuar...")

        if total:
            print(
                f"\nResultado final: "
                f"{puntuacion}/{total} "
                f"({puntuacion / total * 100:.1f}%)"
            )

    def mostrar_dashboard(self):
        """Muestra un dashboard con las estadísticas acumuladas."""
        gestor = GestorEstadisticas()
        datos = gestor.obtener_estadisticas()

        print("\n" + "=" * 70)
        print("                    DASHBOARD DE ESTADÍSTICAS")
        print("=" * 70)

        if not datos["registros"]:
            print("\nNo hay datos registrados aún. ¡Juega algunos desafíos!")
            return

        # Resumen general
        total_registros = len(datos["registros"])
        correctas_total = sum(1 for r in datos["registros"] if r["correcto"])
        porcentaje_general = (correctas_total / total_registros) * 100
        tiempo_total = sum(r["tiempo_segundos"] for r in datos["registros"])

        print(f"\n--- RESUMEN GENERAL ---")
        print(f"Total de ejercicios: {total_registros}")
        print(f"Correctas: {correctas_total} ({porcentaje_general:.1f}%)")
        print(f"Tiempo total: {tiempo_total:.1f} segundos ({tiempo_total/60:.1f} minutos)")
        print(f"Tiempo promedio por ejercicio: {tiempo_total/total_registros:.2f} segundos")

        # Por tema
        print(f"\n--- ESTADÍSTICAS POR TEMA ---")
        temas = {}
        for r in datos["registros"]:
            tema = r["tema"]
            if tema not in temas:
                temas[tema] = {"total": 0, "correctas": 0, "tiempo": 0}
            temas[tema]["total"] += 1
            if r["correcto"]:
                temas[tema]["correctas"] += 1
            temas[tema]["tiempo"] += r["tiempo_segundos"]

        for tema, stats in temas.items():
            porc = (stats["correctas"] / stats["total"]) * 100
            tiempo_prom = stats["tiempo"] / stats["total"]
            print(f"\n{tema}:")
            print(f"  Ejercicios: {stats['total']}")
            print(f"  Correctas: {stats['correctas']} ({porc:.1f}%)")
            print(f"  Tiempo promedio: {tiempo_prom:.2f} segundos")

        # Detalle por tipo de ejercicio
        print(f"\n--- DETALLE POR TIPO DE EJERCICIO ---")
        resumen = datos["resumen"]

        # Ordenar por cantidad de ejercicios
        ordenado = sorted(resumen.items(), key=lambda x: x[1]["total"], reverse=True)

        for clave, info in ordenado[:10]:  # Mostrar top 10
            print(f"\n{info['tema']} - {info['nombre']}:")
            print(f"  Total: {info['total']}")
            print(f"  Correctas: {info['correctas']} ({info['porcentaje']:.1f}%)")
            print(f"  Tiempo promedio: {info['tiempo_promedio']:.2f}s")
            print(f"  Tiempo mediana: {info['tiempo_mediana']:.2f}s")

        print("\n" + "=" * 70)


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    calculator = Calculator()
    simulator = Simulator()
    challenge = Challenge()

    while True:

        print("\n")
        print("=" * 60)
        print("           SIMULADOR DE CINEMÁTICA")
        print("=" * 60)
        print("Temas: MRU | MRUA | Parabólico | Circular")
        print()
        print("1. Simulación y gráficas")
        print("2. Calculadora")
        print("3. Modo desafío")
        print("4. Ver estadísticas")
        print("0. Salir")
        print("=" * 60)

        opcion = input("Selecciona una opción: ").strip()

        if opcion == "1":
            simulator.menu()

        elif opcion == "2":
            calculator.menu()

        elif opcion == "3":
            challenge.menu()

        elif opcion == "4":
            challenge.mostrar_dashboard()

        elif opcion == "0":
            print("\n¡Hasta luego!")
            break

        else:
            print("\nOpción inválida. Intenta de nuevo.")


if __name__ == "__main__":
    main()
