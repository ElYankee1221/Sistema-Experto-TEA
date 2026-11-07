
% 1. Base de Conocimiento: Criterios de falla segun el manual M-CHAT-R
% En los items 2, 5 y 12, contestar con 'si' se interpreta como falla
es_falla(item2, si).
es_falla(item5, si).
es_falla(item12, si).

% En los otros 17 items, responder 'no' se interpreta como falla
es_falla(Item, no) :-
    Item \= item2,
    Item \= item5,
    Item \= item12.

% 2. Regla Recursiva de Extraccion de Fallas
% Caso base: si ya no hay respuestas por evaluar, la lista de fallas esta vacia.
obtener_fallas([],[]).

% Caso 1: La respuesta actual es una falla (el niño no paso el item).
% Se guarda el 'Id' en la cabeza de la lista de fallas [Id|RestoFallas].
obtener_fallas([item(Id, Respuesta)|Resto], [Id|RestoFallas]) :-
    es_falla(Id, Respuesta),
    !,
    obtener_fallas(Resto, RestoFallas).

% Caso 2: La respuesta actual no es una falla (el niño paso el item).
% Se descarta la cabeza de la lista con '_' y se continua analizando el resto.
obtener_fallas([_|Resto], RestoFallas) :-
    obtener_fallas(Resto, RestoFallas).

% 3. Escala de Clasificacion de Riesgo (Etapa 1 M-CHAT-R)
clasificar_riesgo(Puntaje, bajo) :- Puntaje =< 2, !.
clasificar_riesgo(Puntaje, medio) :- Puntaje >= 3, Puntaje =< 7, !.
clasificar_riesgo(Puntaje, alto) :- Puntaje >= 8.

% 4. Regla Principa de Inferencia
% Recibe ListaRespuestas y devuelve: Puntaje total, Nivel de Riesgo y Lista de Items fallidos.
evaluar_paciente(ListaRespuestas, Puntaje, Riesgo, ListaFallas) :-
    obtener_fallas(ListaRespuestas, ListaFallas),
    length(ListaFallas, Puntaje),
    clasificar_riesgo(Puntaje, Riesgo).

    



