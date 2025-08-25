from crewai import Crew, Task, Agent, Process
from crewai.tools import BaseTool
from crewai_tools import CSVSearchTool
from dotenv import load_dotenv, find_dotenv
import os
import json 
from datetime import datetime, timedelta
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

llm=ChatOpenAI(model="gpt-4.1-mini", temperature=0)

csv_imoveis = CSVSearchTool(csv="./files/imoveis.csv", llm=llm, name="Busca de Imóveis", description="Use esta ferramenta para buscar imóveis com base em critérios como preço, localização, número de quartos, banheiros, metragem e tipo de imóvel.")

#agente corretor de imóveis
corretor_imoveis = Agent(
    name="Corretor de Imóveis",
    role="Você é um corretor de imóveis experiente, especializado em ajudar clientes a encontrar a casa dos seus sonhos com base em suas preferências e orçamento.",
    goal="Obtenha as preferências do cliente e critérios para definição e busque imóveis compatíveis com o perfil do clientes no banco de dados.",
    backstory="""
    Especialista de mercado imobiliário, encontra as melhores opções baseadas no perfil do cliente.
    """,
    verbose=True,
    max_iter=5,
    tools=[csv_imoveis],
    aLLow_delegation=False,
    memory=True,
    model="gpt-4.1-mini",
    temperature=0.7
)

buscar_imoveis = Task(
    name="Buscar Imóveis",
    description="Pesquise imóveis na região desejada pelo cliente, considerando faixa de preço e tipo de imóvel",
    expected_output="Lista de imóveis disponíveis com detalhes sobre localização, preço e características",
    agent=corretor_imoveis
)

def obter_precos_imoveis(cidade: str = "geral"):
    precos = {
        "São Paulo": { "tendência": "aumento", "percentual": 5.2},
        "Rio de Janeiro": {"tendência": "estavel", "percentual": 0.0},
        "Belo Horizonte": { "tendência": "queda", "percentual": -3.1},
        "geral": {"tendência": "aumento", "percentual": 4.0}
    }
    return precos.get(cidade, precos["geral"])

class TendenciaPrecosImoveisTool(BaseTool):
    name: str = "Analisador de Preços Imobiliários"
    description: str = "Obtém tendências de preços de imóveis com base na cidade especificada"

    def _run(self, cidade: str) -> dict:
        #Executa a análise de preços imobiliarios e retorna a tendência com base na cidade
        try:
            return obter_precos_imoveis(cidade)
        except Exception as e:
            return {"erro" : f"Erro ao obter tendências de preços {str(e)}"}

analista_mercado = Agent(
    name="Analista de Mercado Imobiliário",
    role="Você é um analista de mercado imobiliário experiente, especializado em coletar dados precisos e relevantes sobre o mercado imobiliário.",
    goal="Obtenha informações atualizadas sobre tendências de preços, demanda e oferta no mercado imobiliário na cidade {cidade}.",
    backstory="""
    Você é um pesquisador de mercado experiente, utiliza dados históricos para prever dados futuros.
    Especializado em coletar dados precisos e relevantes sobre o mercado imobiliário.
    """,
    verbose=True,
    max_iter=5,
    aLLow_delegation=False,
    memory=True,
)

obter_tendencias = Task(
    description="""
    Analise o histórico de preços de imóveis na cidade {cidade} e forneça insights sobre
    valorização ou desvalorização. Considere o tipo de imóvel {tipo_imovel} e a 
    faixa de preço {faixa_preco}
    """,
    expected_output="Resumo das tendências dos preços no mercado imobiliário.",
    tools=[TendenciaPrecosImoveisTool()],
    agent=analista_mercado,
    parameters=["cidade"]
)

analista_noticias = Agent (
    name="Analista de Notícias Imobiliárias",
    role="Você é um analista de notícias imobiliárias experiente, especializado em coletar e resumir notícias relevantes sobre o mercado imobiliário.",
    goal="Obtenha as notícias importantes mais recentes e relevantes sobre o mercado imobiliário na cidade {cidade}.",
    backstory="""
    Você é um pesquisador de mercado experiente, especializado em coletar dados precisos e relevantes sobre o mercado imobiliário.
    Analisr notícias e tendências econômicas que afetam os preços dos imóveis.
    """,
    verbose=True,
    max_iter=5,
    aLLow_delegation=False,
    memory=True,
)

searchTool = DuckDuckGoSearchResults(backend="news", num_results=5)

buscar_noticias = Task(
    description=f"Pesquise notícias recentes sobre o mercado imobiliário. Data atual: {datetime.now()}",
    expected_output="Resumo das principais notícias e tendências imobiliárias",
    agent=analista_noticias,
    tool=[searchTool]
)

consultor_financeiro= Agent (
    name="Consultor Financeiro",
    role="Consultor financeiro",
    goal="Analisa as opções de financiamento imobiliário com base no perfil do cliente.",
    backstory="""
    Você é um consultor financeiro experiente, especializado em ajudar clientes a entender suas opções de financiamento imobiliário.
    Analisa taxas de juros, prazos e condições de diferentes instituições financeiras.
    """,
    verbose=True,
    max_iter=5,
    aLLow_delegation=False,
    memory=True,
)

calcular_financiamento = Task (
    description="Analise a renda do cliente e sugira opções de financiamento viáveis.",
    expected_output="Tabela comparativa com diferentes financiamentos, taxa de juros e prazos.",
    agent=consultor_financeiro, 
)

redator = Agent(
    name="Redator de relatórios",
    role="Redator de relatórios imobiliários",
    goal="Gera um relatório completo e persuasivo com base nas análises de mercado e imóveis encontrados.",
    backstory="""
    Você é um redator experiente, especialista em comunicação. 
    especializado em criar relatórios claros e convincentes.
    Seu trabalho é compilar todas as informações em um formato fácil de entender para o cliente.
    """,
    verbose=True,
    max_iter=5,
    aLLow_delegation=False,
    memory=True,
)

gerar_relatorio = Task(
    description="Gere um relatório detalhado sobre o melhor imóvel encontrado, considerando preços, tendências e financiamento",
    expected_output="Relatório formatado com resumo do mercado, opções recomendadas e justificativa da escolha.",
    agent=redator,
    context=[buscar_imoveis, obter_tendencias, buscar_noticias, calcular_financiamento]
)

imobiliaria_crew = Crew(
    name="Agente Imobiliário",
    description="Agente imobiliário completo, especializado em encontrar imóveis, analisar o mercado e gerar relatórios detalhados para os clientes.",
    agents=[corretor_imoveis, analista_mercado, analista_noticias, consultor_financeiro, redator],
    tasks=[buscar_imoveis, obter_tendencias, buscar_noticias, calcular_financiamento, gerar_relatorio],
    process=Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    verbose=True,
    max_iter=15,
)

result = imobiliaria_crew.kickoff(inputs={"cidade": "Rio de Janeiro", "tipo_imovel": "Apartamento","faixa_preco": "500000-700000"})

print(result.raw)
