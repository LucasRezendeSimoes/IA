import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

# Variáveis de Entrada (Antecedent)
comer = ctrl.Antecedent(np.arange(0, 20, 1), 'comer')
exerc = ctrl.Antecedent(np.arange(0, 10, 1), 'exercicio')

# Variável de saída (Consequent)
peso = ctrl.Consequent(np.arange(0, 15, 1), 'peso')

# Funções de pertinência escolhidas
# "comer" → Triangular
comer['pouco'] = fuzz.trimf(comer.universe, [0, 0, 10])
comer['razoavel'] = fuzz.trimf(comer.universe, [5, 10, 15])
comer['muito'] = fuzz.trimf(comer.universe, [10, 20, 20])

# "exercicio" → Gaussiana
exerc['pouco'] = fuzz.gaussmf(exerc.universe, 2, 1.5)
exerc['razoavel'] = fuzz.gaussmf(exerc.universe, 5, 1.5)
exerc['muito'] = fuzz.gaussmf(exerc.universe, 8, 1.5)

# "peso" → Trapezoidal
peso['magro'] = fuzz.trapmf(peso.universe, [0, 2, 4, 6])
peso['razoavel'] = fuzz.trapmf(peso.universe, [5, 6, 8, 9])
peso['gordo'] = fuzz.trapmf(peso.universe, [8, 10, 13, 15])

# Criando regras
regra_1 = ctrl.Rule(comer['pouco'] & exerc['muito'], peso['magro'])
regra_2 = ctrl.Rule(comer['muito'] & exerc['pouco'], peso['gordo'])
regra_3 = ctrl.Rule(comer['muito'] & exerc['muito'], peso['razoavel'])
regra_4 = ctrl.Rule(comer['pouco'] & exerc['pouco'], peso['magro'])

controlador = ctrl.ControlSystem([regra_1, regra_2, regra_3, regra_4])

# Simulação
CalculoPeso = ctrl.ControlSystemSimulation(controlador)

notaComida = int(input('Quanto você come? (0 a 20): '))
notaExerc = int(input('Quanto você treina? (0 a 10): '))
CalculoPeso.input['comer'] = notaComida
CalculoPeso.input['exercicio'] = notaExerc
CalculoPeso.compute()

valorPeso = CalculoPeso.output['peso']

print("\nCome %d calorias \nGasta %d calorias \nPesa %5.2f Kilos" %(
    notaComida * 100,
    notaExerc * 100,
    valorPeso * 10))

# Criando os 3 gráficos de pertinência
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

comer.view(ax=axes[0])
axes[0].set_title("Função de Pertinência - Comer (Triangular)")

exerc.view(ax=axes[1])
axes[1].set_title("Função de Pertinência - Exercício (Gaussiana)")

peso.view(ax=axes[2])
axes[2].set_title("Função de Pertinência - Peso (Trapezoidal)")

plt.tight_layout()
plt.show()
