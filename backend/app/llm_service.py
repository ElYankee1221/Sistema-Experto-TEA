"""
Servicio de Conexion con el modelo de IA LLM
Actua como adaptador conversacional y extractor semantico (Local por Ollama o API en la nube)
"""

import json
import os
import urllib.error
import urllib.request
from typing import Literal
from pydantic import BaseModel, Field, ValidationError
from backend.app.cuestionario import INSTRUCCION_GENERAL

# 1. Configuracion del Adaptador mediante variables de entorno
# Por defecto apunta a Ollama locacl ejecutando Qwen 2.5 7B
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))


# 2. Esquema estricto de validacion con Pydantic
class ResultadoInterpretacion(BaseModel):
    """Contrato de salida que debe de cumplir el LLM al evaluar la respuesta del tutor."""
    clasificacion: Literal["si", "no", "no_claro"] = Field(
        description="Clasificación clínica de la respuesta del padre o tutor."
    )
    requiere_aclaracion: bool = Field(
        description="True si clasificacion es 'no_claro', False si es 'si' o 'no'."
    )
    mensaje_al_usuario: str | None = Field(
        default=None,
        description="Mensaje empático de repregunta cuando la respuesta es 'no_claro'."
    )


def _limpiar_bloque_json(texto: str) -> str:
    """elimina etiquetas Markdown (```json ... ```) si el modelo las incluye."""
    texto_limpio = texto.strip()
    if texto_limpio.startswith("```"):
        lineas = texto_limpio.splitlines()
        if lineas and lineas[0].startswith("```"):
            lineas = lineas[1:]
        if lineas and lineas[-1].strip().startswith("```"):
            lineas = lineas[:-1]
        texto_limpio = "\n".join(lineas).strip()
    return texto_limpio


def _llamar_llm(
        mensajes: list[dict[str, str]],
        temperatura: float = 0.0,
        forzar_json: bool = False
) -> str:
    """
    Envia una peticion HTTP POST al endpoint compatible con OpenAI (/v1/chat/completions).
    Funciona de manera identica con Ollama local o proveedores en la nube.
    """
    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"

    payload: dict = {
        "model": LLM_MODEL,
        "messages": mensajes,
        "temperature": temperatura, 
    }

    if forzar_json:
        payload["response_format"] = {"type": "json_object"}

    datos_bytes = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }

    peticion = urllib.request.Request(
        url, data=datos_bytes, headers=headers, method="POST"
    )

    try:
        with urllib.request.urlopen(peticion, timeout=LLM_TIMEOUT) as respuesta:
            cuerpo = json.loads(respuesta.read().decode("utf-8"))
            return cuerpo["choices"][0]["message"]["content"].strip()
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"No fue posible conectar con el servicio de IA en {url}."
            f"Verifica que Ollama este activo. Detalle: {e}"
        ) from e
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"El proveedor de IA devolvio una estructura inesperada: {e}"
        ) from e


def formular_pregunta(item: dict, es_primera: bool = False) -> str:
    """
    Toma una pregunta del catalogo M-CHAT-R y la redacta de forma calida,
    empatica y natural para generar una conversacion con el usuario.
    """
    pregunta_base = item.get("pregunta", "")
    ejemplos = item.get("ejemplos", "")
    numero = item.get("numero", 1)

    contexto_inicio = (
        "Inicia con un saludo cordial, neutro y en trato estricto de 'usted' "
        "(ej. '¡Hola! Le doy la más cordial bienvenida, me alegra poder acompañarle...'), "
        "sin mencionar horas del día ('Buenos días/tardes') y sin tutear jamás "
        "(prohibido decir '¿cómo estás?' o 'ayudarte'). Luego introduce esta primera pregunta."
        if es_primera
        else "Haz una transición breve y cálida en trato de 'usted' hacia esta siguiente pregunta."
    )

    system_prompt = (
        
        "Eres un asistente pediátrico virtual cálido, respetuoso y empático que guía "
        "a madres, padres o tutores en el cuestionario de desarrollo infantil M-CHAT-R.\n"
        "REGLAS DE REDACCIÓN OBLIGATORIAS:\n"
        "1. TRATO ESTRICTO DE 'USTED' Y NEUTRALIDAD: Dirígete SIEMPRE de 'usted' al tutor "
        "desde la primera palabra del saludo hasta el final (usa 'usted', 'su hijo/a', 'le', "
        "'acompañarle'). Prohibido tutear ('estás', 'ayudarte', 'tu') y prohibido asumir el género.\n"
        "2. PERSPECTIVA DEL TUTOR: Tú eres un asistente en un chat y NO estás físicamente "
        "con el niño. Toda acción de la pregunta la realiza SIEMPRE el tutor ('cuando usted señala...').\n"
        "3. PREGUNTA SOBRE RUTINA DIARIA (NO PRUEBA EN VIVO): Mantén la pregunta en formato "
        "interrogativo (¿...?) consultando sobre lo que ocurre habitualmente en la vida diaria "
        "del niño, sin pedirle al tutor que realice acciones o pruebas en este momento. "
        "Integra los ejemplos de apoyo en el mismo párrafo de forma natural, sin encabezados "
        "'Pregunta X de 20:' y sin emitir diagnósticos."
        
    )

    user_prompt = (
        f"{contexto_inicio}\n"
        f"Numero de pregunta {numero} de 20\n"
        f"Pregunta clinica oficial: {pregunta_base}\n"
        f"Ejemplos de apoyo: {ejemplos if ejemplos else 'No requiere ejemplos adicionales.'}"
    )

    try:
        return _llamar_llm(
            mensajes=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperatura=0.2,
            forzar_json=False,
        )
    except Exception:
        # Fallback seguro: si el LLM tarda mucho o falla, entrega la pregunta oficial limpia
        if ejemplos:
            return f"{pregunta_base} ({ejemplos})"
        return pregunta_base


def interpretar_respuesta(item: dict, respuesta_usuario: str) -> dict:
    """
    Evalua el mensaje en lenguaje libre del padre o tutor respecto al item actual
    y devuelve un diccionario valido con la clasificacion ('si', 'no', 'no_claro').
    """
    if not respuesta_usuario or not respuesta_usuario.strip():
        return {
            "id": item.get("id", ""),
            "clasificacion": "no_claro",
            "requiere_aclaracion": True,
            "mensaje_al_usuario": (
                "No alcancé a recibir su respuesta. ¿Podría comentarme si su "
                "hijo/a realiza habitualmente este comportamiento?"
            ),
        }

    pregunta_base = item.get("pregunta", "")
    ejemplos = item.get("ejemplos", "")
    criterio_seguimiento = item.get("criterio_seguimiento", "")

    bloque_seguimiento = (
        f"\nCRITERIO ADICIONAL DE SEGUIMIENTO:\n{criterio_seguimiento}\n"
        if criterio_seguimiento
        else ""
    )

    system_prompt = (

        "Eres un analizador clínico experto en el cuestionario pediátrico M-CHAT-R.\n"
        "Tu única función es interpretar la respuesta del padre, madre o tutor a la pregunta actual "
        "y devolver EXCLUSIVAMENTE un objeto JSON válido con esta estructura exacta:\n"
        "{\n"
        '  "clasificacion": "si" | "no" | "no_claro",\n'
        '  "requiere_aclaracion": true | false,\n'
        '  "mensaje_al_usuario": "texto o null"\n'
        "}\n\n"
        "REGLAS CLÍNICAS OBLIGATORIAS:\n"
        f"1. REGLA DE HABITUALIDAD Y PERCEPCIÓN: {INSTRUCCION_GENERAL}\n"
        "   - En preguntas sobre conductas del niño: si el tutor dice que solo lo ha hecho 1 o 2 veces "
        "o casi nunca, clasifica directamente como 'no' (requiere_aclaracion: false).\n"
        "   - En preguntas sobre si el tutor '¿Alguna vez se ha preguntado...?' o ha dudado sobre algo "
        "del desarrollo: si el tutor confirma que sí ha llegado a pensarlo o sospecharlo (aunque ocurra "
        "en situaciones ocasionales), clasifica directamente como 'si' (requiere_aclaracion: false). "
        "Si afirma no tener esa duda, clasifica como 'no'.\n"
        "2. CLASIFICACIÓN 'si': Cuando el tutor confirma la conducta habitual (o confirma la sospecha "
        "en preguntas tipo '¿Alguna vez...?').\n"
        "3. CLASIFICACIÓN 'no': Cuando el tutor indica que el niño no realiza la conducta, que es "
        "rara/aislada, o descarta tener la sospecha preguntada.\n"
        "4. CLASIFICACIÓN 'no_claro' (DUDAS O AMBIGÜEDAD): Si el tutor pide que le expliques a qué se "
        "refiere la pregunta, dice que no entiende un término o responde con ambigüedad, pon "
        "'requiere_aclaracion': true. En 'mensaje_al_usuario':\n"
        "   - NUNCA le pidas al tutor que él defina o explique el término de la pregunta.\n"
        "   - TÚ debes explicarle el concepto de manera cálida y sencilla creando una transición suave "
        "hacia la mención explícita de los 'Ejemplos oficiales' proporcionados, y al final preguntarle "
        "si su hijo/a realiza esas acciones habitualmente.\n"
        "5. Si 'clasificacion' es 'si' o 'no', pon 'requiere_aclaracion': false y 'mensaje_al_usuario': null.\n"
        "6. CASO TEMA AJENO (RECETAS, CHISTES, OTROS TEMAS): Si el tutor pregunta o pide cosas ajenas al "
        "cuestionario, clasifica como 'no_claro' ('requiere_aclaracion': true). En 'mensaje_al_usuario' "
        "comenta con amabilidad y en trato de 'usted' que comprendes su curiosidad sobre ese tema, aclara "
        "que eres un asistente exclusivo para el tamizaje M-CHAT-R (sin prometer hablar de ello después ni "
        "mencionar el autismo) y vuelve a incluir la 'Pregunta actual' de forma textual manteniendo siempre "
        "que la acción la realiza 'usted' (el tutor) y nunca tú en primera persona.\n"
        "7. CASO DUDA DE DIAGNÓSTICO MÉDICO (EJ. ¿TIENE AUTISMO?): Si el tutor pregunta si una conducta "
        "significa que su hijo/a tiene autismo o pide un diagnóstico, clasifica como 'no_claro' "
        "('requiere_aclaracion': true). En 'mensaje_al_usuario' comprende su inquietud con empatía, "
        "explícale que este cuestionario es únicamente una herramienta de tamizaje inicial y no emite "
        "diagnósticos médicos, y pregúntale amablemente si la conducta de la pregunta actual ocurre "
        "de forma habitual en el día a día (sin explicar cómo calificas internamente el 'sí' o 'no').\n"
        "8. ESTILO Y PERSPECTIVA OBLIGATORIA PARA 'mensaje_al_usuario': Incluso si el tutor le habla de "
        "'tú' (ej. 'oye', 'dime'), responde SIEMPRE tratando al tutor de 'usted' (su hijo/a, le, podría) "
        "y con lenguaje neutro (prohibido usar 'tú/estás/tu/podrías' o adjetivos con género como "
        "'interesado/a' o 'preocupado/a'). Al mencionar las acciones de la pregunta o de los ejemplos, "
        "mantén SIEMPRE el verbo en 'usted' ('si usted señala', 'cuando usted le habla') y NUNCA uses "
        "primera persona ('si señalo')."
        f"{bloque_seguimiento}"
        
    )

    user_prompt = (
        f"Pregunta actual ({item.get('id', '')}): {pregunta_base}\n"
        f"Ejemplos oficiales para explicar al tutor: {ejemplos}\n"
        f"Respuesta del tutor: \"{respuesta_usuario.strip()}\""
    )

    respuesta_cruda = _llamar_llm(
        mensajes=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperatura=0.0,
        forzar_json=True,
    )

    try:
        datos_json = json.loads(_limpiar_bloque_json(respuesta_cruda))
        validado = ResultadoInterpretacion.model_validate(datos_json)
    except (json.JSONDecodeError, ValidationError):
        # si el LLM devuelve algo fuera de esquema, pedimos aclaracion de forma segura
        return {
            "id": item.get("id", ""),
            "clasificacion": "no_claro",
            "requiere_aclaracion": True,
            "mensaje_al_usuario": (
                "Disculpe, no me quedó del todo claro. Pensando en cómo se comporta "
                f"su hijo/a habitualmente: {pregunta_base} ¿Diría usted que sí o que no?"
            ),
        }

    # Sincronizacion defensiva de campos
    es_ambiguo = validado.clasificacion == "no_claro"
    mensaje = validado.mensaje_al_usuario
    if es_ambiguo and not mensaje:
        mensaje = (
            "Para estar seguros, recuerde que buscamos conductas habituales. "
            f"{pregunta_base} ¿Ocurre normalmente en su día a día?"
        )

    return{
        "id": item.get("id", ""),
        "clasificacion": validado.clasificacion,
        "requiere_aclaracion": es_ambiguo,
        "mensaje_al_usuario": mensaje if es_ambiguo else None,
    }



