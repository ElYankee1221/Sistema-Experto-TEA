"""
Servicio de Conexion con SWI-Prolog 
Traduce las respuestas de Python a estructuras logicas de Prolog y ejecuta la inferencia
"""

from pathlib import Path
import threading
from pyswip import Prolog
from backend.app.cuestionario import CUESTIONARIO_MCHAT

# 1. Solucion dinamica para la ruta hacia prolog/motor_mchat.pl
RUTA_BASE = Path(__file__).resolve().parent.parent.parent
RUTA_MOTOR_PROLOG = RUTA_BASE / "prolog" / "motor_mchat.pl"

# Conjunto con los 20 IDs autorizados
IDS_VALIDOS = {item["id"] for item in CUESTIONARIO_MCHAT}

# 2. Instancia global de Prolog y candado de seguridad para los hilos (Mutex Lock)
_prolog = Prolog()
_prolog_lock = threading.Lock()
_motor_cargado = False


def _asegurar_motor_cargado() -> None:
    """Carga el archivo motor_prolog.pl una sola vez en la memoria de SWI-Prolog."""
    global _motor_cargado
    if not _motor_cargado:
        if not RUTA_MOTOR_PROLOG.exists():
            raise FileNotFoundError(
                f"No se encontro el archivo de Prolog en: {RUTA_MOTOR_PROLOG}"
            )
        # .as_posix() invierte las barras de las rutas en windows '\'
        # para evitar problemas de interpretacion como caracter de escape con SWI-Prolog
        _prolog.consult(RUTA_MOTOR_PROLOG.as_posix()) 
        _motor_cargado = True


def formatear_respuestas_prolog(respuestas: list[dict[str, str]]) -> str:
    """
    Valida la itegridad de los datos de una lista de diccionarios de Python:
        [{"id": "item1", "respuesta": "si"}, {"id": "item2", "respuesta": "si"}]
    Al formato de lista de estructuras que se espera para SWI-Prolog:
        "[item(tiem1, si), item(item2, si)]"
    """
    elementos = []
    ids_procesados = set()

    for item in respuestas:
        item_id = item.get("id", "").strip().lower()
        resp = item.get("respuesta", "").strip().lower()

        # 1. Validar de su existencia dentro del catalogo oficial
        if item_id not in IDS_VALIDOS:
            raise ValueError(
                f"Item no reconocido: '{item_id}'. Solo se admiten preguntas de item1 a item20."
            )

        # 2. Validacion de duplicados
        if item_id in ids_procesados:
            raise ValueError(
                f"Item duplicado detectado: '{item_id}'."
            )
        ids_procesados.add(item_id)

        # 3. Validacion de valor clinico del item
        if resp not in ("si", "no"):
            raise ValueError(
                f"Respuesta invalida '{resp}' en {item_id}. Solo se permite 'si' o 'no'."
            )

        elementos.append(f"item({item_id}, {resp})")

    return "[" + ", ".join(elementos) + "]"


def obtener_recomendacion_clinica(riesgo: str) -> str:
    """Devuelve la conducta clinica calculada segun el manual oficial M-CHAT-R."""
    recomendaciones = {
        "bajo": (
            "Bajo riesgo (0 a 2 puntos). Si el niño es menor de 24 meses, "
            "repetir el M-CHAT-R a los 24 meses. Continuar con la vigilancia "
            "del desarrollo en las visitas de control del niño sano."
        ),
        "medio": (
            "Riesgo medio (3 a 7 puntos). Se recomienda administrar la "
            "entrevista de seguimiento (M-CHAT-R/F) sobre los ítems fallidos "
            "para obtener informacion adicional o acudir a revisión pediátrica."
        ),
        "alto": (
            "Riesgo alto (8 a 20 puntos). Se recomienda remitir el caso de "
            "inmediato para evaluación diagnóstica integral y determinar la "
            "necesidad de intervención temprana."
        )
    }
    return recomendaciones.get(riesgo, "Nivel de riesgo no reconocido.")


def evaluar_cuestionario(respuestas: list[dict[str, str]]) -> dict:
    """
    Ejecuta la regla 'evaluar_paciente' en SWI-Prolog de forma segura
    y retorna un diccionari nativo de Python listo para enviarse en JSON.
    """
    if not respuestas:
        raise ValueError("La lista de respuestas no puede estar vacía.")

    lista_prolog = formatear_respuestas_prolog(respuestas)
    consulta = f"evaluar_paciente({lista_prolog}, Puntaje, Riesgo, ListaFallas)"

    with _prolog_lock:
        _asegurar_motor_cargado()
        soluciones = list(_prolog.query(consulta))

    if not soluciones:
        raise RuntimeError("El motor de Prolog no devolvío ninguna solución.")

    resultado = soluciones[0]
    puntaje = int(resultado["Puntaje"])
    riesgo = str(resultado["Riesgo"])
    items_fallados = [str(falla) for falla in resultado["ListaFallas"]]

    return {
        "puntaje": puntaje,
        "riesgo": riesgo,
        "items_fallados": items_fallados,
        "recomendacion": obtener_recomendacion_clinica(riesgo)
    }


 


