# importar bibliotecas
import pandas as pd
import numpy as np
import os
from scipy import optimize

# Verifica a pasta corrente
pasta = os.getcwd()

# Junta caminho corrente + pasta com os arquivos "ambos", "homens", "mulheres"
pasta_ambos = os.path.join(pasta, "ambos")
#pasta_mulheres = os.path.join(pasta, "mulheres")
#pasta_homens = os.path.join(pasta, "homens")

# Lista arquivos das pastas
arquivos_ambos = os.listdir(pasta_ambos)
#arquivos_mulheres = os.listdir(pasta_mulheres)
#arquivos_homens = os.listdir(pasta_homens)

# Lista os arquivos somente excel e que comecem com 'ambos', 'homens', 'mulheres'
arq_ambos_xls = [arq_ambos for arq_ambos in arquivos_ambos if arq_ambos[-3:]=='xls']
#arq_mulher_xls = [arq_mulher for arq_mulher in arquivos_mulheres if arq_mulher[-3:]=='xls']
#arq_homem_xls = [arq_homem for arq_homem in arquivos_homens if arq_homem[-3:]=='xls']

len(arq_ambos_xls)

# Inicializa o dataframe vazio
df_ambos = pd.DataFrame()

# Dataframe ambos
pular_linhas = list(range(0,5)) + list(range(46,62)) + list(range(103,113))
colunas = ['x', 'qx_mil', 'dx', 'lx', 'Lx', 'Tx', 'Ex']

for arq_ambos in arq_ambos_xls:
    ano = arq_ambos[-8:-4]
    dados=pd.read_excel(os.path.join(pasta_ambos, arq_ambos),
                        names = colunas,
                        skiprows = pular_linhas,
                        usecols = "A:G").assign(Ano=ano)
    df_ambos = df_ambos.append(dados)

df_ambos.info()

# Retirar sinais na idade 80 anos e colocar 80
# Tratamento da idade 80+ para os anos.
df_ambos[df_ambos['x'].astype(str).str.contains("80")].head()

# Tratamento da idade 80+ para os anos.
df_ambos.loc[80,'x'] = 80
# Aproveitar e corrigir o qx aos 80 anos de alguns arquivos
df_ambos.loc[80,'qx_mil'] = 1000.0

df_ambos[df_ambos['x'].astype(str).str.contains("80")].head()

# Para classificar por ano
#linha_anterior = sorted(df_ambos.loc[79,:].values.tolist(), key=lambda x: x[7])
#novas_instancias = sorted(df_ambos.loc[80,:].values.tolist(), key = lambda x: x[7])
# sorted --> subslistas classificadas pelo atributo "Ano" --> x[7]
df_to_lista = sorted(df_ambos.values.tolist(), key = lambda x: x[7])

# FUNÇÕES ATUARIAIS

def comutacao(FAj):
    qx_add = []
    qx_add_temp = []
    lx = []
    dx = []
    Lx = []
    Tx = []
    expx = []
    
    fx = 0.5
    #FAj = 100.0
    lx.append(100000.0)
    id_inicio = 0
    w = 120 # idade limite, onde não haverá vivos
    qx_add_temp.append(qx)
    qx_add = [*qx_add_temp[0]]
    
    for idade in range(id_inicio, 80):
        # primeiro passo
        dx.append(qx[idade]*lx[idade]/1000.0)  # dx = qx[idade]*lx[idade]/1000.0
        # segundo passo
        lx.append(lx[idade]-dx[idade])  # lx[idade+1] = lx[idade] - dx[idade]
    
    for idade in range(80, w):
        if lx[idade] != 0.0:
            # terceiro passo
            #lx[idade+2] = lx[idade+1]*(lx[idade+1]/(lx[idade]+FAj))
            lx.append(lx[idade]*(lx[idade]/(lx[idade-1]+FAj)))
        else:
            w = idade
            break        
    
    for idade in range(80, w):
        # quarto passo
        #dx[idade]=lx[idade]-lx[idade+1]
        dx.append(lx[idade]-lx[idade+1])
        # quinto passo
        # qx[idade] = dx[idade]/lx[idade]
        qx_add.append(dx[idade]/lx[idade]*1000.0)

    for idade in range(id_inicio, w):
        # sexto passo.
        # Lx[idade] = fx*lx[idade] + (1 - fx)*lx[idade+1]
        Lx.append(fx*lx[idade] + (1 - fx)*lx[idade+1])
    
    for idade in range(id_inicio, w):
        if lx[idade] != 0.0:
            # setimo passo
            # Tx = sum(Lx[idade:])
            Tx.append(sum(Lx[idade:]))
            # oitavo passo
            # exp = Tx[idade]/lx[idade]
            expx.append(Tx[idade]/lx[idade])
        else:
            break

    return idade, qx_add, dx, lx, Lx, Tx, expx


def comutacao_res(FAj):
    qx_add = []
    qx_add_temp = []
    lx = []
    dx = []
    Lx = []
    Tx = []
    expx = []
    
    fx = 0.5
    #FAj = 100.0
    lx.append(100000.0)
    id_inicio = 0
    w = 120 # idade limite, onde não haverá vivos
    qx_add_temp.append(qx)
    qx_add = [*qx_add_temp[0]]


    for idade in range(id_inicio, 80):
        # primeiro passo
        dx.append(qx[idade]*lx[idade]/1000.0)  # dx = qx[idade]*lx[idade]/1000.0
        # segundo passo
        lx.append(lx[idade]-dx[idade])  # lx[idade+1] = lx[idade] - dx[idade]
    
    for idade in range(80, w):
        if lx[idade] != 0.0:
            # terceiro passo
            #lx[idade+2] = lx[idade+1]*(lx[idade+1]/(lx[idade]+FAj))
            lx.append(lx[idade]*(lx[idade]/(lx[idade-1]+FAj)))
        else:
            w = idade
            break        
    
    for idade in range(80, w):
        # quarto passo
        #dx[idade]=lx[idade]-lx[idade+1]
        dx.append(lx[idade]-lx[idade+1])
        # quinto passo
        # qx[idade] = dx[idade]/lx[idade]
        qx_add.append(dx[idade]/lx[idade]*1000.0)

    for idade in range(id_inicio, w):
        # sexto passo.
        # Lx[idade] = fx*lx[idade] + (1 - fx)*lx[idade+1]
        Lx.append(fx*lx[idade] + (1 - fx)*lx[idade+1])
    
    for idade in range(id_inicio, w):
        if lx[idade] != 0.0:
            # setimo passo
            # Tx = sum(Lx[idade:])
            Tx.append(sum(Lx[idade:]))
            # oitavo passo
            # exp = Tx[idade]/lx[idade]
            expx.append(Tx[idade]/lx[idade])
        else:
            break

    erro = expx[79] - targetx
    return abs(erro)

# Função para unir as listas em linha
def unirSeries(df, explode):
    idx = df.index.repeat(df[explode[0]].str.len())
    df1 = pd.concat([
        pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
    df1.index = idx

    return df1.join(df.drop(explode, 1), how='left')


dados = []
vetor_fatores = []
fatores_lista = []
vetor_dados = []
dados_lista = [] # np.empty((8,0)).tolist()
ano = 1997
inicio = 0
idade = []
ano_rept = []

for i in range(0,21): # Leitura de cada arquivo desde 1997
    ano += 1
    # intervalos de cada tabela de dados para cada ano
    fim = 81*(i+1)
    dados = df_to_lista[inicio:fim]
    inicio = fim
    FAj = 100.0
    
    # qx das tabelas do IBGE até 79 anos
    qx = [item[1] for item in dados][:-1]
    targetx = dados[79][6] # Expectativa de vida aos 80, na tabela do IBGE

    sol = optimize.minimize_scalar(comutacao_res) # , args=(FAj) se existissem mais outros parametros, usar args

    #sol.fun # erro / #sol.x # fator / #sol.sucess # sucesso /  #sol.nit # numero de iterações
    vetor_fatores = [str(ano), sol.x, sol.nit, sol.fun, sol.success]
    # aplicar append, pois é uma lista de uma lista de valores por vez
    fatores_lista.append(vetor_fatores) 

    x, qx_add, dx, lx, Lx, Tx, expx = comutacao(sol.x)

    idade = np.arange(0, x+1).tolist() #list(range(0, idade+1))
    ano_rept = np.repeat(ano, x+1).tolist()

    vetor_dados = [idade, qx_add, dx, lx[:-1], Lx, Tx, expx, ano_rept]
    # Aplicar extend para gravar as comutações, pois é uma lista de listas de valores
    dados_lista.append(vetor_dados)
#    df_dados_lista = df_dados_lista.append(pd.DataFrame(dados_lista))

# Salva no Dataframel
df_fatores = pd.DataFrame(fatores_lista, columns=['ano', 'fator_ajuste', 'num_interacoes', 'erro', 'converge'])
#df_dados = df_dados.transpose()
#df_dados.columns = list(df_ambos)
df_temp = pd.DataFrame(dados_lista, columns=['idade', 'qx_mil', 'dx', 'lx', 'Lx', 'Tx', 'expx', 'ano'])
df_dados = unirSeries(df_temp,['idade', 'qx_mil', 'dx', 'lx', 'Lx', 'Tx', 'expx', 'ano'])
df_dados = df_dados.reset_index(drop=True) 


