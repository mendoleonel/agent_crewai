#######################
# Agente para fazer análise de mercado
# Autor: Mendo Leonel
# Data: 2024-06-26
#
#     Pesquisador de Mercado: Coleta dados sobre o mercado, concorrentes e tendências.
#     Analista de tendências: Interpreta os dados coletados e gera insights acionáveis
#.    Redator do relatório: Compila os insights em um relatório claro e conciso.
#######################

import os
from dotenv import load_dotenv, find_dotenv
from crewai import Agent, Process, Crew, Task

load_dotenv(find_dotenv())


pesquisador = Agent(
    role="Pesquisador de Mercado",
    goal="Coletar e organizar informações relevantes sobre {sector}",
    backstory="""
    Você é um pesquisador de mercado experiente, especializado em coletar dados precisos e relevantes sobre o {sector}.
    Seu trabalho é garantir que todas as informações estejam atualizadas e bem documentadas.
    """,
    name="Pesquisador de Mercado",
    allow_delegation=False,
    verbose=True,
    model="gpt-4.1-mini",
    temperature=0.7,
)

analista = Agent(
    name="Analista de tendências",
    role="Analista de tendências de mercado",
    goal="Interpretar os dados coletados do {sector} e identificar padrões, oportunidades e gerar insights acionáveis",
    backstory="""
    Você é um analista de tendências de mercado, especializado em interpretar dados complexos e identificar padrões emergentes, oportunidades
    e ameaças no {sector}.
    """,
    allow_delegation=False,
    verbose=True,
    model="gpt-4.1-mini",
    temperature=0.7,
)

redator = Agent(
    name="Redator do relatório",
    role="Redator de relatórios de análise de mercado",
    goal="Compilar os insights gerados pelo analista em um relatório consolidado, claro e conciso",
    backstory="""
    Você é um redator experiente, especializado em criar relatórios claros e concisos que comunicam insights complexos de maneira acessível.
    """,
    allow_delegation=False,
    verbose=True,
    model="gpt-4.1-mini",
    temperature=0.7,
)

coleta_dados = Task(
    description=(
        "1. Pesquisar e coletar informações atualizadas sobre {sector}"
        "2. Organizar os dados em categorias relevantes"
        "3. Identificar os principais players, tendências e estatísticas do {sector}."
    ),
    expected_output="Um documento estruturado contendo dados de mercado sobre o {sector}",
    agent=pesquisador
)

analise_tendencias = Task(
    description=(
        "1. Examinar os dados coletados pelo Pesquisador de Mercado"
        "2. Identificar padrões, oportunidades e ameaças no {sector}"
        "3. Elaborar uma análise detalhada destacando os principais pontos."
    ),
    expected_output="Um relatório com insights e tendências acionáveis baseados na análise dos dados do {sector}",
    agent=analista
)

redacao_relatorio = Task(
    description=(
        "1. Usar a análise de tendências para criar um relatório detalhado sobre o {sector}"
        "2. Estruturar o relatório de forma clara e concisa"
        "3. Garantir que o relatório seja acessível para todos os stakeholders."
    ),
    expected_output="Um relatório final consolidado, claro e conciso sobre a análise de mercado do {sector} em formato markdown pronto para leitura e apresentação.",
    agent=redator
)

analise_mercado_crew = Crew(
    agents=[pesquisador, analista, redator],
    tasks=[coleta_dados, analise_tendencias, redacao_relatorio],
    process=Process.sequential,
    verbose=True
)

resultado = analise_mercado_crew.kickoff(inputs={"sector": "Agentes de IA"})

print(resultado)