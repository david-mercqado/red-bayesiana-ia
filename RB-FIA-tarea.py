# -*- coding: utf-8 -*-
!pip install -q pgmpy

"""
Red Bayesiana: Diagnóstico Clínico de Enfermedad Respiratoria
Implementación basada en pgmpy con Inferencia por Eliminación de Variables (Variable Elimination)
"""

try:
    from pgmpy.models import DiscreteBayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork as DiscreteBayesianNetwork

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def construir_red():
    # 1. Definición del Grafo Dirigido Acíclico (DAG)
    modelo = DiscreteBayesianNetwork([
        ('Lluvia', 'Resfrio'),
        ('Resfrio', 'Tos'),
        ('Resfrio', 'Fiebre'),
        ('Tos', 'Enfermedad'),
        ('Fiebre', 'Enfermedad')
    ])

    # 2. Tablas de Probabilidad Condicional (CPD)

    # P(Lluvia)
    cpd_lluvia = TabularCPD(
        variable='Lluvia',
        variable_card=2,
        values=[[0.70],   # No
                [0.30]],  # Si
        state_names={'Lluvia': ['No', 'Si']}
    )

    # P(Resfrio | Lluvia)
    cpd_resfrio = TabularCPD(
        variable='Resfrio',
        variable_card=2,
        values=[[0.85, 0.40],   # No
                [0.15, 0.60]],  # Si
        evidence=['Lluvia'],
        evidence_card=[2],
        state_names={'Resfrio': ['No', 'Si'], 'Lluvia': ['No', 'Si']}
    )

    # P(Tos | Resfrio)
    cpd_tos = TabularCPD(
        variable='Tos',
        variable_card=2,
        values=[[0.80, 0.20],   # No
                [0.20, 0.80]],  # Si
        evidence=['Resfrio'],
        evidence_card=[2],
        state_names={'Tos': ['No', 'Si'], 'Resfrio': ['No', 'Si']}
    )

    # P(Fiebre | Resfrio)
    cpd_fiebre = TabularCPD(
        variable='Fiebre',
        variable_card=2,
        values=[[0.90, 0.30],   # No
                [0.10, 0.70]],  # Si
        evidence=['Resfrio'],
        evidence_card=[2],
        state_names={'Fiebre': ['No', 'Si'], 'Resfrio': ['No', 'Si']}
    )

    # P(Enfermedad | Tos, Fiebre)
    cpd_enfermedad = TabularCPD(
        variable='Enfermedad',
        variable_card=2,
        values=[
            # [Tos=No, F=No], [Tos=No, F=Si], [Tos=Si, F=No], [Tos=Si, F=Si]
            [0.98, 0.60, 0.50, 0.10],   # No
            [0.02, 0.40, 0.50, 0.90]    # Si
        ],
        evidence=['Tos', 'Fiebre'],
        evidence_card=[2, 2],
        state_names={
            'Enfermedad': ['No', 'Si'],
            'Tos': ['No', 'Si'],
            'Fiebre': ['No', 'Si']
        }
    )

    # 3. Asignación y verificación del modelo
    modelo.add_cpds(cpd_lluvia, cpd_resfrio, cpd_tos, cpd_fiebre, cpd_enfermedad)
    assert modelo.check_model(), "El modelo tiene inconsistencias."
    return modelo


def consultar_red(inferencia, variable, evidencia=None, titulo="Consulta"):
    print("\n" + "=" * 50)
    print(f"{titulo}")
    print("=" * 50)
    resultado = inferencia.query(
        variables=[variable],
        evidence=evidencia if evidencia else {}
    )
    print(resultado)
    return resultado


if __name__ == "__main__":
    red = construir_red()
    inferencia = VariableElimination(red)

    print("--- RED BAYESIANA: DIAGNÓSTICO MÉDICO ---")

    # Caso 1: Probabilidad a priori
    consultar_red(inferencia, variable="Enfermedad", titulo="Caso 1: P(Enfermedad) a priori")

    # Caso 2: Diagnóstico con síntomas observados
    consultar_red(
        inferencia,
        variable="Enfermedad",
        evidencia={"Tos": "Si", "Fiebre": "Si"},
        titulo="Caso 2: P(Enfermedad | Tos=Si, Fiebre=Si)"
    )

    # Caso 3: Inferencia diagnóstica inversa
    consultar_red(
        inferencia,
        variable="Lluvia",
        evidencia={"Tos": "Si"},
        titulo="Caso 3: P(Lluvia | Tos=Si)"
    )