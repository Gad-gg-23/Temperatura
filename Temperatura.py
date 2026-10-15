"""Punto de entrada único de la práctica de predicción de temperatura (ESCOM).

Subcomandos:
    datos      -> descarga el histórico meteorológico (obtener_datos.py)
    entrenar   -> entrena y compara los modelos (entrenar_modelo.py)
    predecir   -> hace una predicción en vivo (predecir.py)

Ejemplos:
    python Temperatura.py datos
    python Temperatura.py entrenar
    python Temperatura.py predecir
    python Temperatura.py predecir --registrar 18.4
"""
import argparse
import sys

import entrenar_modelo
import obtener_datos
import predecir

sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="comando", required=True)

    sub.add_parser("datos", help="Descarga el histórico meteorológico")
    sub.add_parser("entrenar", help="Entrena y compara los modelos")
    parser_predecir = sub.add_parser("predecir", help="Predicción en vivo")
    parser_predecir.add_argument("--registrar", type=float, default=None, help="Lectura real del termómetro en ESCOM")

    args = parser.parse_args()

    if args.comando == "datos":
        obtener_datos.main()
    elif args.comando == "entrenar":
        entrenar_modelo.main()
    elif args.comando == "predecir":
        predecir.predecir_actual(args.registrar)


if __name__ == "__main__":
    main()
