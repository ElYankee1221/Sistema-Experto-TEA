"""
Servidor Principal y Maquina de Estados (main.py)
Orquesta las sesiones del cuestionario M-CHAT-R, el servicio LLM y el motor SWI-Prolog.
"""

from typing import Literal
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.cuestionario import obtener_item_por_indice, total_preguntas
from backend.app.llm_service import formular_pregunta, interpretar_respuesta
from backend.app.prolog_service import evaluar_cuestionario

# Limite de intentos para respuestas ('no_claro') consecutivas antes de solicitar una respuesta directa
MAX_INTENTOS_POR_PREGUNTA = 2

app = FastAPI(
    title="API Sistema Experto M-CHAT-R (TEA)",
    description=(
        "Backend orquestador para el tamizaje temprano de TEA. " \
        "Gestiona el flujo conversacional con IA y la evaluación lógica en SWI-Prolog."
    ),
    version="1.0.0",
)

# 1. C onfiguracion de CORS para permitir la conexion desde cualquier puerto del Frontend local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Almacen de sesiones en memoria (Maquina de Estados por session_id)
SESIONES: dict[str, dict] = {}

# -----------------------------------------------------
# ESQUEMAS PYDANTIC (Contrato de Datos con el Frontend)
# -----------------------------------------------------

class PeticionIniciarSesion(BaseModel):
    """Datos opcionales para iniciar o reiniciar una evaluacion."""
    session_id: str | None = Field(
        default=None,
        description="ID de sesion opcional. Si no se envia, el backend genera un UUID automaticamente.",
        examples=["sesion_paciente_01"],
    )

class PeticionChat(BaseModel):
    """Estructura JSON que el Frontend envia a cada turno del chat."""
    session_id: str = Field(
        ...,
        description="Indentificador unico de la sesion activa.",
        examples=["sesion_paciente_01"],
    )
    mensaje: str = Field(
        ...,
        description="Texto libre escrito por el padre, madre o tutor.",
        examples=["Si, cada vez que señalo un juguete voltea a verlo."],
    )

class ReporteClinico(BaseModel):
    """Estructura del resultado final generado por SWI-Prolog al completar las 20 preguntas."""
    puntaje: int = Field(description="Total de items con falla clinica (0 a 20).")
    riesgo: Literal["bajo", "medio", "alto"] = Field(description="Nivel de riego calculado.")
    items_fallados: list[str] = Field(description="Lista de identificadores de items con falla.")
    recomendacion: str = Field(description="Conducta clinica sugerida segun el manual M-CHAT-R.")

class RespuestaChat(BaseModel):
    """Estructura JSON estandarizada que el Backend devuelve siempre al Frontend."""
    session_id: str = Field(description="Identificador de la sesion.")
    estado: Literal["en_proceso", "completado"] = Field(
        description="Indica si el cuestionario sigue activo o ya concluyo."
    )
    pregunta_actual: int = Field(
        description="Numero de pregunta actual (1 a 20)"
    )
    progreso: str = Field(
        description="Texto de avance listo para mostrar en la interfaz (ej. '1/20')."
    )
    mensaje_bot: str = Field(
        description="Respuesta o pregunta redactada por el asistente llm para mostrar en el chat."
    )
    requiere_aclaracion: bool = Field(
        default=False,
        description="True si la respuesta anterior no fue clara y se sigue en la misma pregunta."
    )
    sugerir_botones_binarios: bool = Field(
        default=False,
        description="True cuando el usuairo acumula varios intentos ambiguos; el Frontend puede mostrar botones 'Si/No'."
    )
    reporte_final: ReporteClinico | None = Field(
        default=None,
        description="Contiene el dictamen de SWI-Prolog unicamente cuando el estado == 'completado'."
    )

# ---------------------------------------------
# FUNCIONES AUXILIARES DE LA MAQUINA DE ESTADOS
# ---------------------------------------------

def _crear_nueva_sesion(session_id: str) -> dict:
    """Inicializa el estado de una sesion en la pregunta 1 (indice 0)."""
    primer_item = obtener_item_por_indice(0)
    if not primer_item:
        raise RuntimeError("El catalogo del cuestionario M-CHAT-R esta vacio.")

    mensaje_inicial = formular_pregunta(primer_item, es_primera=True)

    nueva_sesion = {
        "session_id": session_id,
        "estado": "en_proceso",
        "indice_actual": 0,
        "intentos_fallidos": 0,
        "respuestas": [],
        "ultimo_mensaje_bot": mensaje_inicial,
        "reporte_final": None,
    }
    SESIONES[session_id] = nueva_sesion
    return nueva_sesion

def _normalizar_atajo_binario(texto: str) -> str | None:
    """
    Si el usurio responde exactamente 'si'/'sí' o 'no' (por ejemplo, usndo botones 
    de ayuda en el frontend), devuelve la clasificacion directa sin consumir tiempo del LLM.
    """
    limpio = texto.strip().lower().rstrip(".!")
    if limpio in ("si", "sí"):
        return "si"
    if limpio == "no":
        return "no"
    return None

# ------------------------
# ENDPOINTS DE LA API REST
# ------------------------

@app.get("/api/health", tags=["Sistema"])
def verificar_salud() -> dict:
    """Verifica que el servidor FastAPI este en linea."""
    return {
        "stratus": "ok",
        "servicio": "Sistema Experto M-CHAT-R",
        "total_preguntas": total_preguntas(),
    }

@app.post("/api/sesion/iniciar", response_model=RespuestaChat, tags=["Cuestionario"])
def iniciar_sesion(peticion: PeticionIniciarSesion | None = None) -> RespuestaChat:
    """
    Crea (o reinicia) una sesion de evaluacion y devuelve el saludo de bienvenida
    junto con la Pregunta 1 formulada por la IA.
    """
    nuevo_id = (
        peticion.session_id.strip()
        if peticion and peticion.session_id and peticion.session_id.strip()
        else str(uuid.uuid4())
    )
    sesion = _crear_nueva_sesion(nuevo_id)
    total = total_preguntas()

    return RespuestaChat(
        session_id=nuevo_id,
        estado="en_proceso",
        pregunta_actual=1,
        progreso=f"1/{total}",
        mensaje_bot=sesion["ultimo_mensaje_bot"],
        requiere_aclaracion=False,
        sugerir_botones_binarios=False,
        reporte_final=None,
    )

@app.post("/api/chat", response_model=RespuestaChat, tags=["Cuestionario"])
def procesar_mensaje_chat(peticion: PeticionChat) -> RespuestaChat:
    """
    Recibe la respuesta del tutor, evalua su clasificacion clinica mediante el LLM,
    avanza al indice del cuestionario o ejecuta la inferencia en SWI-Prolog al terminar.
    """
    session_id = peticion.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="El campo 'session_id' no puede estar vacío.")

    total = total_preguntas()

    # Si la sesion no existe aun, la inicializamos y entregamos la primera pregunta
    if session_id not in SESIONES:
        sesion = _crear_nueva_sesion(session_id)
        return RespuestaChat(
            session_id=session_id,
            estado="en_proceso",
            pregunta_actual=1,
            progreso=f"1/{total}",
            mensaje_bot=sesion["ultimo_mensaje_bot"],
            requiere_aclaracion=False,
            sugerir_botones_binarios=False,
            reporte_final=None,
        )

    sesion = SESIONES[session_id]

    # Si el cuestionario ya habia finalizado, devolvemos el reporte almacenado
    if sesion["estado"] == "completado":
        return RespuestaChat(
            session_id=session_id,
            estado="completado",
            pregunta_actual=total,
            progreso=f"{total}/{total}",
            mensaje_bot=(
                "La evaluación M-CHAT-R ya ha concluido. A continuación puede consultar "
                "el reporte de resultados o iniciar una nueva sesión."
            ),
            requiere_aclaracion=False,
            sugerir_botones_binarios=False,
            reporte_final=sesion["reporte_final"],
        )

    indice_actual = sesion["indice_actual"]
    item_actual = obtener_item_por_indice(indice_actual)
    if not item_actual:
        raise HTTPException(status_code=500, detail="Índice de pregunta fuera de rango.")

    # 1. Evaluar respuesta (atajo binario directo o interpretacion semantica con LLM)
    atajo = _normalizar_atajo_binario(peticion.mensaje)
    if atajo:
        resultado_llm = {
            "id": item_actual["id"],
            "clasificacion": atajo,
            "requiere_aclaracion": False,
            "mensaje_al_usuario": None,
        }
    else:
        try:
            resultado_llm = interpretar_respuesta(item_actual, peticion.mensaje)
        except ConnectionError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Servicio de IA temporalmente no disponible: {e}",
            ) from e

    clasificacion = resultado_llm["clasificacion"]

    #2. Bifurcacion A: La respuesta no fue clara ('no_claro') -> permanecemos en el mismo item
    if clasificacion == "no_claro":
        sesion["intentos_fallidos"] += 1
        sugerir_botones = sesion["intentos_fallidos"] >= MAX_INTENTOS_POR_PREGUNTA

        if sugerir_botones:
            mensaje_aclaracion = (
                f"{resultado_llm['mensaje_al_usuario'] or ''}\n\n"
                "Para poder avanzar con la siguiente pregunta del tamizaje, "
                f"¿podría indicarme concretamente con un 'Sí' o un 'No' si su hijo/a "
                f"realiza esta conducta habitualmente? ({item_actual['pregunta']})"
            ).strip()
        else:
            mensaje_aclaracion = resultado_llm["mensaje_al_usuario"]

        sesion["ultimo_mensaje_bot"] = mensaje_aclaracion

        return RespuestaChat(
            session_id=session_id,
            estado="en_proceso",
            pregunta_actual=item_actual["numero"],
            progreso=f"{item_actual['numero']}/{total}",
            mensaje_bot=mensaje_aclaracion,
            requiere_aclaracion=True,
            sugerir_botones_binarios=sugerir_botones,
            reporte_final=None,
        )

    # 3. Bifurcacion B: Respuesta clara ('si', 'no') -> guardamos hecho y avanzamos indice
    sesion["respuestas"].append({
        "id": item_actual["id"],
        "respuesta": clasificacion,
    })
    sesion["intentos_fallidos"] = 0
    sesion["indice_actual"] += 1

    siguiente_item = obtener_item_por_indice(sesion["indice_actual"])

    # 4. Si aun quedan preguntas por responder (indcies 1 al 19)
    if siguiente_item is not None:
        nueva_pregunta_bot = formular_pregunta(siguiente_item, es_primera=False)
        sesion["ultimo_mensaje_bot"] = nueva_pregunta_bot

        return RespuestaChat(
            session_id=session_id,
            estado="en_proceso",
            pregunta_actual=siguiente_item["numero"],
            progreso=f"{siguiente_item['numero']}/{total}",
            mensaje_bot=nueva_pregunta_bot,
            requiere_aclaracion=False,
            sugerir_botones_binarios=False,
            reporte_final=None,
        )

    # 5. Si ya se respondieron las 20 preguntas -> Invcamos el motor logico en SWI-Prolog
    try:
        dictamen_prolog = evaluar_cuestionario(sesion["respuestas"])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al ejecutar el motor de inferencia en Prolog: {e}",
        ) from e

    sesion["estado"] = "completado"
    sesion["reporte_final"] = dictamen_prolog
    mensaje_cierre = (
        "Muchas gracias por completar las 20 preguntas del cuestionario M-CHAT-R. "
        "Sus respuestas han sido analizadas por el sistema experto. "
        f"Resultado obtenido: riesgo {dictamen_prolog['riesgo'].upper()} "
        f"(puntaje: {dictamen_prolog['puntaje']} de {total}). "
        f"{dictamen_prolog['recomendacion']}"
    )
    sesion["ultimo_mensaje_bot"] = mensaje_cierre

    return RespuestaChat(
        session_id=session_id,
        estado="completado",
        pregunta_actual=total,
        progreso=f"{total}/{total}",
        mensaje_bot=mensaje_cierre,
        requiere_aclaracion=False,
        sugerir_botones_binarios=False,
        reporte_final=dictamen_prolog,
    )