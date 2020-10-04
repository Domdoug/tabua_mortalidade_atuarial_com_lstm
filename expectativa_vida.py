# importar bibliotecas
import pandas as pd
import numpy as np
import os
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import time

# Verifica a pasta corrente
pasta = os.getcwd()

# Junta caminho corrente + pasta com os arquivos "ambos", "homens", "mulheres"
pasta_ambos = os.path.join(pasta, "ambos")
pasta_mulheres = os.path.join(pasta, "mulheres")
pasta_homens = os.path.join(pasta, "homens")

# Lista arquivos das pastas
arquivos_ambos = os.listdir(pasta_ambos)
arquivos_mulheres = os.listdir(pasta_mulheres)
arquivos_homens = os.listdir(pasta_homens)

# Lista os arquivos somente excel e que comecem com 'ambos', 'homens', 'mulheres'
arq_ambos_xls = [arq_ambos for arq_ambos in arquivos_ambos if arq_ambos[-3:]=='xls']
arq_mulher_xls = [arq_mulher for arq_mulher in arquivos_mulheres if arq_mulher[-3:]=='xls']
arq_homem_xls = [arq_homem for arq_homem in arquivos_homens if arq_homem[-3:]=='xls']

#len(arq_ambos_xls)

# ==================== INICIO FEMININO =================

# Pular linhas na leitura dos arquivos
pular_linhas1 = list(range(0,5)) + list(range(46,62)) + list(range(103,113))
pular_linhas2 = list(range(0,5)) + list(range(46,61)) + list(range(102,113))
pular_linhas3 = list(range(0,4)) + list(range(45,61)) + list(range(102,113))

colunas = ['x', 'qx_mil', 'dx', 'lx', 'Lx', 'Tx', 'Ex']

# # Inicializa o dataframe vazio
df_mulher = pd.DataFrame()
for arq_mulher in arq_mulher_xls:
    ano = arq_mulher[-8:-4]
    if ano in ['1998', '1999']: 
        pular_linhas = pular_linhas2
    elif ano in ['2003']:
        pular_linhas = pular_linhas3
    else:
        pular_linhas = pular_linhas1
    dados=pd.read_excel(os.path.join(pasta_mulheres, arq_mulher),
                        names = colunas,
                        skiprows = pular_linhas,
                        usecols = "A:G").assign(Ano=ano)
    df_mulher = df_mulher.append(dados)

# Tratamento da idade 80+ para os anos.
df_mulher.loc[80,'x'] = 80
# Aproveitar e corrigir o qx aos 80 anos de alguns arquivos
df_mulher.loc[80,'qx_mil'] = 1000.0

# ==================== FIM FEMININO =================
# ==================== INICIO MASCULINO =================
# Inicializa o dataframe vazio
df_homem = pd.DataFrame()
# Dataframe ambos
pular_linhas1 = list(range(0,5)) + list(range(46,62)) + list(range(103,113))
pular_linhas2 = list(range(0,5)) + list(range(46,61)) + list(range(102,113))
colunas = ['x', 'qx_mil', 'dx', 'lx', 'Lx', 'Tx', 'Ex']

for arq_homem in arq_homem_xls:
    ano = arq_homem[-8:-4]
    if ano in ['1998', '1999']:
        pular_linhas = pular_linhas2
    else:
        pular_linhas = pular_linhas1
    
    dados=pd.read_excel(os.path.join(pasta_homens, arq_homem),
                        names = colunas,
                        skiprows = pular_linhas,
                        usecols = "A:G").assign(Ano=ano)
    df_homem = df_homem.append(dados)

# Tratamento da idade 80+ para os anos.
df_homem.loc[80,'x'] = 80
# Aproveitar e corrigir o qx aos 80 anos de alguns arquivos
df_homem.loc[80,'qx_mil'] = 1000.0

# ==================== FIM MASCULINO =================
# ========================== INICIO AMBOS ========================
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

# VERIFICACAO: df_ambos_add = pd.DataFrame()
df_ambos_add = df_ambos[80:81]
df_ambos_add
# ======================FIM AMBOS============================

# =============== GRAFICOS INICIO =====================
# Grafico de lx
# agrega os dataframes, identificando todos
frames = [df_mulher, df_homem, df_ambos]
df_agregado = pd.concat(frames, keys=['mulher', 'homem', 'ambos'], names=['sexo', 'IdLinha']).reset_index()
df_agregado = df_agregado.query("x != 80")
df_agregado['qx_prob'] = df_agregado['qx_mil']/1000.0

# grafico de sobreviventes - lx
sns.set_style("darkgrid")
graf = sns.FacetGrid(df_agregado, col="sexo", hue="Ano")
graf.map(sns.lineplot, "x", "lx")
graf.add_legend()
graf.set_xlabels("idade")
graf.set_ylabels("Sobreviventes")

# grafico - probabilidade de morte
graf = sns.FacetGrid(df_agregado, col="sexo", hue="Ano")
graf.map(sns.lineplot, "x", "qx_prob")
graf.add_legend()
graf.set_xlabels("idade")
graf.set_ylabels("Prob. Morte")

# Grafico de Expecativa de vida ao nascer
# Dataframe com x = 0
#sns.set() #reset o seaborn

df_agregado_0 = df_agregado.query("x == 0")
graf_0 = sns.lineplot(x='Ano', y='Ex', data=df_agregado_0, hue="sexo")
graf_0.set_title("Expectativa de vida ao nascer")
graf_0.set_ylabel('Idade')
plt.xticks(rotation=45)
graf_0.plot()




# =============== GRAFICOS FIM =====================
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

# Leitura de cada arquivo desde 1998
dados = []
vetor_fatores = []
fatores_lista = []
vetor_dados = []
dados_lista = [] # np.empty((8,0)).tolist()
ano = 1997
inicio = 0
idade = []
ano_rept = []
w = []

for i in range(0,21): 
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
    # grava as features básicas da tábua de mortalidade
    x, qx_add, dx, lx, Lx, Tx, expx = comutacao(sol.x)
    w.append(x)
     
    idade = np.arange(0, x+1).tolist() #list(range(0, idade+1))
    ano_rept = np.repeat(ano, x+1).tolist()

    vetor_dados = [idade, qx_add, dx, lx[:-1], Lx, Tx, expx, ano_rept]
    # Aplicar extend para gravar as comutações, pois é uma lista de listas de valores
    dados_lista.append(vetor_dados)
#    df_dados_lista = df_dados_lista.append(pd.DataFrame(dados_lista))

# Salva no Dataframel
w_max = max(w)
df_fatores = pd.DataFrame(fatores_lista, columns=['ano', 'fator_ajuste', 'num_interacoes', 'erro', 'converge'])
#df_dados = df_dados.transpose()
#df_dados.columns = list(df_ambos)
# df_temp: df_temp.shape -> 21,8. vetor das variáveis em cada linha 
df_temp = pd.DataFrame(dados_lista, columns=['idade', 'qx_mil', 'dx', 'lx', 'Lx', 'Tx', 'expx', 'ano'])
df_dados = unirSeries(df_temp,['idade', 'qx_mil', 'dx', 'lx', 'Lx', 'Tx', 'expx', 'ano'])
# df_dados.shape -> (2421,8). Desfeito vetor. Variaveis ao longo das linhas
df_dados = df_dados.reset_index(drop=True)


# ============== INICIO GRAFICOS =======================
# grafico de sobreviventes - lx
sns.set_style("darkgrid")
graf = sns.lineplot(data=df_dados, x="idade", y="lx", hue="ano")
graf.set_title("Quantidade de vidas")
#graf.set_xlabel("idade")
graf.set_ylabel("Sobreviventes")

# grafico - probabilidade de morte
graf = sns.lineplot(data=df_dados, x="idade", y="qx_mil", hue="ano")
graf.set_title("Quantidade de mortos")
ylabels = ['{:,.3f}'.format(y) for y in graf.get_xticks()/100]
graf.set_yticklabels(ylabels)
graf.set_ylabel("Prob. Morte")

# Grafico de Expecativa de vida ao nascer
# Dataframe com x = 0
# eixo y:expectativa de vida ao nascer. idade=0. todos os anos (eixo x)

df_dadosx_0 = df_dados.query("idade == 0")
df_dadosx_0['ano'] = df_dadosx_0['ano'].astype(str)
graf_0 = sns.lineplot(x='ano', y='expx', data=df_dadosx_0)
graf_0.set_title("Expectativa de vida ao nascer")
graf_0.set_ylabel('Expectativa vida em anos')
plt.xticks(rotation=45)
plt.show()

# verificar qx
df_dadosx_0 = df_dados.query("idade == 0")
df_dadosx_0['ano'] = df_dadosx_0['ano'].astype(str)
graf_0 = sns.lineplot(x='ano', y='qx_mil', data=df_dadosx_0)
graf_0.set_title("Probabilidade de morte de 1998- 2018")
graf_0.set_ylabel('Probabilidade de morte')
plt.xticks(rotation=45)
plt.show()

# ============== FIM GRAFICOS =======================

# Preparar o DataFrame para o LSTM
#lista_temp = df_temp.values
#lista_temp = df_temp.values.tolist()
#tamanho = [len(n) for n in lista_temp[)][1]] # tamanho de cada sublista
#menor = min(tamanho) # menor valor entre as sublistas
df_lstm = df_dados[['ano','idade','qx_mil', 'lx']].copy()
# deletar as linhas onde idade >=113 (menor tamanho)
indexNames = df_lstm[df_lstm['idade']>=113].index
df_lstm.drop(indexNames, inplace=True)
df_lstm = df_lstm.reset_index(drop=True)
df_lstm['xt'] = df_lstm['ano'].astype(str) + '_' + df_lstm['idade'].astype(str)
df_lstm['qx_prob'] = df_lstm['qx_mil']/1000.0

# ====================== GRAFICO ========================
# # Gráfico temporal matplotlib. Seaborn demora muito a renderizar
#plt.style.use('seaborn-whitegrid')

ax = plt.axes()
ax.plot('xt','qx_prob', data=df_lstm)
ax.set_title('Probabilidade de morte - Periodo 1999-2018')
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.set_xlabel('Ano_idade')
ax.set_ylabel('Prob. Morte')
ax.grid(True)

# Gráfico genérico para o artigo
df_dados2018 = df_dados.query("ano == 2018")
#df_dadosx_0['ano'] = df_dadosx_0['ano'].astype(str)
#graf_0 = sns.lineplot('x', 'lx', ci=None, data=df_dados2018)
plt.plot('idade', 'lx', data=df_dados2018) 
plt.title("Funçao de sobrevivência")
plt.xlabel('Anos vividos')
plt.ylabel('População')
#plt.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
#plt.xticks(rotation=45)
#plt.xticks([])
#plt.yticks([])
# plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.xaxis.set_ticklabels([])
plt.gca().axes.yaxis.set_ticklabels([])
plt.savefig('grafico_exp3.png')
plt.show()




# ====================== FIM DO GRAFICO ========================

# Persistence Model Forecast: BASELINE
# implementar

# Obs.: line plot of the test dataset (blue) compared to the
# predicted values (orange) is also created showing
# the persistence model forecast in context

# ==========================================
#a.) convert time series into supervised learning problem
# frame a sequence as a supervised learning problem

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

#b.) create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)

#c.) transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw values
    raw_values = series.values
    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    # rescale values to -1, 1
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test

#d.) Define the LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1]))
    #model.add(Dense(y.shape[1], activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=["accuracy"])
    # fit network
    for _ in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
        model.reset_states()
    return model

#e.) forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]
 
#f.)  evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
    forecasts = list()
    for i in range(len(test)):
        X, _ = test[i, 0:n_lag], test[i, n_lag:]
        # make forecast
        forecast = forecast_lstm(model, X, n_batch)
        # store the forecast
        forecasts.append(forecast)
    return forecasts
 
#g.) invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted
 
#h.) inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = np.array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted
 
#i.) evaluate the model with RMSE 
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i+1), rmse))
 
#j.) plot the forecasts
def plot_forecasts(series, forecasts, n_test):
    # plot the entire dataset in blue
    plt.plot(series.values)
    # plot the forecasts in red
    for i in range(len(forecasts)):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        plt.plot(xaxis, yaxis, color='red')
 # show the plot
    plt.show()
    
# load the dataset
series = df_lstm['qx_prob'].copy()  # series = read_csv('sales_year.csv', usecols=[1], engine='python')# configure
#series = df_lstm['qx_prob'].copy()

# inicio do cronometro do processamento
start = time.time()
n_lag = 113 # 1 # 113 corresponde a uma tábua(idade de 0 a 113. No caso um ano.
n_seq = 10 # 5 # 3 # número de anos adiante
n_test = 113 # 791 # Agora simulacao com 33 % teste = 791 (7 anos) / 113 => corresponde ao ano de 2018 como teste # 10
n_epochs = 30 # 1500
n_batch = 1
n_neurons = 50 #50

#prepare data
scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)

#fit model
model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)

#forecasts
forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)

#inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
actual = [row[n_lag:] for row in test]
actual = inverse_transform(series, actual, scaler, n_test+2)

#evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)

#plot forecasts
plot_forecasts(series, forecasts, n_test+2)
# fim do cronometro do processamento
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print()
print('Tempo de processamento:')
print('{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))
print()


# imprimir somente um range de featues futuras
colunas = list(range(2019, 2029))
df_forecasts = pd.DataFrame(forecasts, columns=colunas)
df_forecasts[[2019, 2022, 2025, 2028]].plot()


# verificar a saida do modelo:
model.summary()
