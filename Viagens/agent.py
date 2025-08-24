#######################
# Agente para planejar uma viagem
# Autor: Mendo Leonel
# Data: 2024-06-26
#
# Descrição: Este script cria dois agentes usando a biblioteca CrewAI.
# O primeiro agente é um planejador de viagens que cria um roteiro detalhado
# para uma viagem à Grécia. O segundo agente é um orçamentista que calcula
# o custo total estimado da viagem, incluindo transporte, hospedagem,
# alimentação e atividades. Os agentes trabalham juntos em uma Crew para
# completar as tarefas sequencialmente.
#
#######################
import os
from dotenv import load_dotenv
from crewai import Agent, Process, Crew, Task

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("A variável de ambiente OPENAI_API_KEY não está definida. Verifique seu arquivo .env.")


planejador_de_viagem = Agent(
    name="Planejador de Viagem",
    role="Você é um planejador de viagens especializado em criar itinerários personalizados para seus clientes com base em suas preferências e interesses.",
    goal="Planejar todos os detalhes de uma viagem para a Grécia, incluindo roteiros e atividades.",
    backstory="Você é um especialista em planejamento de viagens, sempre em busca de novas aventuras e experiências. Seu objetivo é garantir que os detalhes da viagem sejam organizados de maneira eficiente e agradável.",
    verbose=True,
    model="gpt-4",
    temperature=0.7,
    max_tokens=1500
)

orcamentista = Agent(
    name="Orçamentista",
    role="Você é um orçamentista especializado em calcular custos e despesas para viagens, garantindo que os planos estejam dentro do orçamento do cliente.",
    goal="Estimar o custo de uma viagem, considerando transporte, hospedagem, alimentação e atividades.",
    backstory="Você é um analista financeiro focado em viagem. Sua missão é garantir que os custos estejam dentro do orçamento, criando estimativas precisas para cada parte da viagem.",
    verbose=True
)


planeja_roteiro = Task(
    description="Criar um roteiro detalhado para uma viagem para Grécia, incluindo as cidades, atividades e transporte",
    agent=planejador_de_viagem,
    expected_output="Um roteiro com a sequencia de cidades, atividades diárias e meios de transporte entre os destinos."
)

estima_orcamento = Task(
    description="Calcular o custo total estimado da viagem, incluindo transporte, hospedagem, alimentação e atividades",
    agent=orcamentista,
    expected_output="Um estimativa de orçamento detalhada com os custos estimados para cada parte da viagem e  o custo total."
)


viagem_crew = Crew(
    agents=[planejador_de_viagem, orcamentista],
    tasks=[planeja_roteiro, estima_orcamento],
    process= Process.sequential
)


result = viagem_crew.kickoff()

print(result)
