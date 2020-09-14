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
from keras.layers import LSTM

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

len(arq_ambos_xls)

# ==================== INICIO FEMININO =================
# # Inicializa o dataframe vazio
df_mulher = pd.DataFrame()

# Dataframe mulher
pular_linhas1 = list(range(0,5)) + list(range(46,62)) + list(range(103,113))
pular_linhas2 = list(range(0,5)) + list(range(46,61)) + list(range(102,113))
pular_linhas3 = list(range(0,4)) + list(range(45,61)) + list(range(102,113))
# 2003: pular_linhas2 = list(range(0,4)) + list(range(45,61)) + list(range(101,113))
colunas = ['x', 'qx_mil', 'dx', 'lx', 'Lx', 'Tx', 'Ex']

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
graf_0 = sns.lineplot('Ano', 'Ex', ci=None, data=df_agregado_0, hue="sexo")
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
graf_0 = sns.lineplot('ano', 'expx', ci=None, data=df_dadosx_0)
graf_0.set_title("Expectativa de vida ao nascer")
graf_0.set_ylabel('Expectativa vida em anos')
plt.xticks(rotation=45)
plt.show()

# verificar qx
df_dadosx_0 = df_dados.query("idade == 0")
df_dadosx_0['ano'] = df_dadosx_0['ano'].astype(str)
graf_0 = sns.lineplot('ano', 'qx_mil', ci=None, data=df_dadosx_0)
graf_0.set_title("Probabilidade de morte de 1998- 2018")
graf_0.set_ylabel('Probabilidade de morte')
plt.xticks(rotation=45)
plt.show()

# ============== FIM GRAFICOS =======================

# Preparar o DaaFrame para o LSTM
#lista_temp = df_temp.values
lista_temp = df_temp.values.tolist()
#tamanho = [len(n) for n in lista_temp[)][1]] # tamanho de cada sublista
#menor = min(tamanho) # menor valor entre as sublistas
df_lstm = df_dados[['ano','idade','qx_mil']].copy()
# deletar as linhas onde idade >=113 (menor tamanho)
indexNames = df_lstm[df_lstm['idade']>=113].index
df_lstm.drop(indexNames, inplace=True)
df_lstm = df_lstm.reset_index(drop=True)
df_lstm['xt'] = df_lstm['ano'].astype(str) + '_' + df_lstm['idade'].astype(str)
df_lstm['qx_prob'] = df_lstm['qx_mil']/1000.0

# # Gráfico temporal matplotlib. Seaborn demora muito a renderizar
#plt.style.use('seaborn-whitegrid')

ax = plt.axes()
ax.plot('xt','qx_prob', data=df_lstm)
ax.set_title('Probabilidade de morte - Periodo 1999-2018')
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.set_xlabel('Ano_idade')
ax.set_ylabel('Prob. Morte')
ax.grid(True)

'''
X = df_lstm.values
X = X.reshape(len(X),1)
train_size = int(len(X)*0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
'''

#df_lstm = colunas_selecionadas.copy().T

#df_lstm.columns=[str(i) for i in range(1998,2019)]

#vetor_lstm = np.array([vt[0:menor] for vt in lista_temp[0]])
#[len(n) for n in vetor_lstm] # verificando
# split dados. Est'formatado de uma forma diferente
#train, test = vetor_lstm[:,:-1], vetor_lstm[:,-1]
#train, test = vetor_lstm[:-1,0:menor+1], vetor_lstm[-1,0:menor+1]



#X = series.values
#train, test = X[0:-12], X[-12:]
# walk-forward validation

# Persistence Model Forecast: BASELINE
# split data into train and test
X = df_lstm['qx_mil'].values
# split dados. Como temos de 1998-2018, A série de 2018, um vetor de 113 posições, será para teste
# 1998-2018: 21 periodoss. 21*113 anos (idade) = df_lsmt.shape[0]
train, test = X[0:-113], X[-113:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# make prediction
	predictions.append(history[-1])
	# observation
	history.append(test[i])
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
# line plot of observed vs predicted

plt.plot(test)
plt.plot(predictions)
plt.show()

# Obs.: line plot of the test dataset (blue) compared to the
# predicted values (orange) is also created showing
# the persistence model forecast in context

# Transform the time series into a supervised learning problem

# repeat experiment

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
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test

#d.) Define the LSTM network 
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam',metric=["accuracy"])
    # fit network
    for i in range(nb_epoch):
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
        X, y = test[i, 0:n_lag], test[i, n_lag:]
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
        forecast = array(forecasts[i])
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
    pyplot.plot(series.values)
    # plot the forecasts in red
    for i in range(len(forecasts)):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        pyplot.plot(xaxis, yaxis, color='red')
 # show the plot
    pyplot.show()
    
# load the dataset
# TROCAR!!!!!!!11 O ARQUIVO SERIES
series = read_csv('sales_year.csv', usecols=[1], engine='python')# configure
n_lag = 1
n_seq = 3
n_test = 10
n_epochs = 1500
n_batch = 1
n_neurons = 50

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




# ==== APAGAR DAQUI PARA BAIXO ==========

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -113]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-113], train[:, -113]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]


# transform data to be stationary
raw_values = df_lstm['qx_mil'].values
diff_values = difference(raw_values, 1)

# transform data to be supervised learning
supervised = series_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:-113], supervised_values[-113:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# repeat experiment
repeats = 30
error_scores = list()
for r in range(repeats):
	# fit the model
	lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
	# forecast the entire training dataset to build up state for forecasting
	train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
	lstm_model.predict(train_reshaped, batch_size=1)
	# walk-forward validation on the test data
	predictions = list()
	for i in range(len(test_scaled)):
		# make one-step forecast
		X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
		yhat = forecast_lstm(lstm_model, 1, X)
		# invert scaling
		yhat = invert_scale(scaler, X, yhat)
		# invert differencing
		yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
		# store forecast
		predictions.append(yhat)
	# report performance
	rmse = sqrt(mean_squared_error(raw_values[-113:], predictions))
	print('%d) Test RMSE: %.3f' % (r+1, rmse))
	error_scores.append(rmse)

# summarize results
results = pd.DataFrame()
results['rmse'] = error_scores
print(results.describe())
results.boxplot()
plt.show()