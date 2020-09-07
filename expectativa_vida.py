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

def comutacao():
    lx = []
    dx = []
    Lx = []
    Tx = []
    exp = []
    fx = 0.5
    FAj = 100.0
    lx.append(100000.0)
    id_inicio = 0
    w = 120 # idade limite, onde não haverá vivos

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
        qx.append(dx[idade]/lx[idade]*1000.0)

    for idade in range(id_inicio, w):
        # sexto passo. PAREI AQUI. OUT OF BOUND
        # Lx[idade] = fx*lx[idade] + (1 - fx)*lx[idade+1]
        Lx.append(fx*lx[idade] + (1 - fx)*lx[idade+1])
    
    for idade in range(id_inicio, w):
        if lx[idade] != 0.0:
            # setimo passo
            # Tx = sum(Lx[idade:])
            Tx.append(sum(Lx[idade:]))
            # oitavo passo
            # exp = Tx[idade]/lx[idade]
            exp.append(Tx[idade]/lx[idade])
        else:
            break

    return qx, dx, lx, Lx, Tx, exp, FAj


# O segundo elemento é o que se espera atingir
def expectResidual(idade, exp_ant, Tx, lx_1, lx_2 ):

    erro = exp[79] - exp_ant
    return abs(erro)

def comutacao_res(FAj):
    lx = []
    dx = []
    Lx = []
    Tx = []
    exp = []
    fx = 0.5
    #FAj = 100.0
    lx.append(100000.0)
    id_inicio = 0
    w = 120 # idade limite, onde não haverá vivos

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
        qx.append(dx[idade]/lx[idade]*1000.0)

    for idade in range(id_inicio, w):
        # sexto passo. PAREI AQUI. OUT OF BOUND
        # Lx[idade] = fx*lx[idade] + (1 - fx)*lx[idade+1]
        Lx.append(fx*lx[idade] + (1 - fx)*lx[idade+1])
    
    for idade in range(id_inicio, w):
        if lx[idade] != 0.0:
            # setimo passo
            # Tx = sum(Lx[idade:])
            Tx.append(sum(Lx[idade:]))
            # oitavo passo
            # exp = Tx[idade]/lx[idade]
            exp.append(Tx[idade]/lx[idade])
        else:
            break

    erro = exp[79] - target
    return abs(erro)




#ano = 2015 para testar 2015, i=17
dados = []
fajs = []
dados_final = pd.DataFrame()
#fx = 0.5
# id_inicio = 0 # para a interpolação. Correção: colocar 0

#colunas = list(df_teste)
ano = 1997
inicio = 0
for i in range(0,21):
    ano += 1
    # intervalos de cada tabela de dados para cada ano
    fim = 81*(i+1)
    dados = df_to_lista[inicio:fim]
    inicio = fim
    
    # FAj = 100.0 # Faz a primeira tentativa100.0 # 231.59552332036
    # qx das tabelas do IBGE até 79 anos
    qx = [item[1] for item in dados][:-1]
    target = dados[79][6] # Expectativa de vida aos 80, na tabela do IBGE

    # for idade in range(id_inicio, w): #colocar um stop de 150 para o range
        # qx = dados[0][1]  # qx até os 79 anos. Tabela fornecida IBGE
        #qx(idade)

        #dx = map(dx,qx) #dx(idade, qx)
        
        #if idade==80:
            # Tx = dados[idade][5]
        #    lx_1 = dados[idade-1][3]
        #    lx_2 = dados[idade-2][3]
        #    lx = dados[idade][3]
        #    lx1 = lx_1 * (lx_1 / lx_2 + FAj)
        #    Lx = fx * lx + (1 - fx) * lx1
            # exp_ant = dados[idade][6]  # Expectativa a ser alcançada 

            
            #exp_calc = Tx / (lx_1 * (lx_1 / lx_2 + FAj))

            # sol = optimize.minimize_scalar(expectResidual, args=(exp_calc, Tx, lx_1, lx_2))
    qx, dx, lx, Lx, Tx, exp, FAj = comutacao()
    
    sol = optimize.minimize_scalar(comutacao_res) # , args=(FAj)

    # qx, dx, lx, Lx, Tx, exp, FAj = comutacao_res(id_inicio)           
            #lx1 = lx * (lx / (lx_1 + FAj))
            #dx = lx - lx1
            #qx_mil = dx / lx * 1000
            #Lx = fx * lx + (1 - fx) * lx1

            #exp = Tx / lx
                
            #valores = [idade, qx_mil, dx, lx, Lx, Tx, exp, str(ano)]
            #dados[idade] = valores

        #else:
        #    lx_2 = dados[idade-2][3]
        #    lx_1 = dados[idade-1][3]

            #lx = lx_1 * (lx_1 / (lx_2 + FAj))
            #lx1 = lx * (lx / (lx_1 + FAj))
            #dx = lx - lx1
            #if lx != 0:
            #    qx_mil = dx / lx * 1000
            #    Lx = fx * lx + (1 - fx) * lx1
            #    Tx = dados[idade-1][5] - dados[idade-1][4]
            #    exp = Tx / lx
            #    valores = [idade, qx_mil, dx, lx, Lx, Tx, exp, str(ano)]
            #else:
            #    qx_mil = 0    
            #    Lx = 0
            #    Tx = 0
            #    exp = 0
            #    valores = [idade, qx_mil, dx, lx, Lx, Tx, exp, str(ano)]
            # grava na lista
            #dados.append(valores)
            #fajs.append[FAj_result]





    # Grava no DataFrame
    dados_final = dados_final.append(dados)





colunas = list(df_ambos)
dados_final.columns = colunas
dados_final.to_excel('arquivo4.xls')
