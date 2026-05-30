:-dynamic tiene/1.
:-dynamic enfermedad/1.

lista([]):-enfermedad(E), write(E).
lista([H|T]):-assert(tiene(H)), lista(T).

test(X) :- limpiar, lista(X).

enfermedad('DENGUE GRAVE'):-tiene(s1),tiene(s9),tiene(s10),tiene(s11).
enfermedad('SINDROME PULMONAR POR HANTAVIRUS'):-tiene(s1),tiene(s6),tiene(s15).
enfermedad('LEPTOSPIROSIS (ENFERMEDAD DE WEIL)'):-tiene(s1),tiene(s7),tiene(s8),tiene(s12).
enfermedad('COMPLICACION NEUROLOGICA POR ZIKA (GUILLAIN-BARRE)'):-tiene(s1),tiene(s4),tiene(s13).
enfermedad('ZIKA'):-tiene(s1),tiene(s4),tiene(s5).
enfermedad('CHIKUNGUNYA'):-tiene(s1),tiene(s3),tiene(s5).
enfermedad('DENGUE CLASICO'):-tiene(s1),tiene(s2),tiene(s5),tiene(s14).

enfermedad('No Determinado (Sin patron claro)').

limpiar:-retract(tiene(_)), fail.
limpiar.