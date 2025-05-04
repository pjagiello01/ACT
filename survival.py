# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Analiza przeżycia na przykładzie zbioru colon z pakietu survival 
# ### Cel projektu:
# Celem projektu jest zaproponowanie prostego schematu analizy przeżycia na podstawie danych dotyczących adiuwantowej chemioterapii w leczeniu osób chorych na raka jelita grubego dostępnych w pakiecie survival w R.
# W ramach projektu zaproponowane zostaną modele:
# 1. Parametryczny
# 2. Nieparametryczny
# 3. Semiparametryczny
# <br><br>
# ### Analiza przeżycia
# Analiza przeżycia skupia się na modelowaniu czasu do zaistnienia jakiegoś zdarzenia, przykładowo takim zdarzeniem może być śmierć pacjenta lub nawrot choroby. Najczęsciej dane opierają się na badaniu prowadzonym przez jakiś określony czas na określonej próbie. Obserwację zdarzenia dla jednostki podczas okresu badania określamy jako failure, a brak obserwacji jako censoring. <br>
# W analizie przeżycia chcemy określić wpływ zmiennych na prawdopodobieństwo zaistnienia danego zdarzenia. Przykładowo możemy chcieć sprawdzić, czy terapia jest efektywna w leczeniu choroby.
# <br><br>
# ### Opis danych
# Jak już zostało wspomniane do projektu użyliśmy danych dotyczących adiuwantowej chemioterapii w leczeniu osób chorych na raka jelita grubego. Dane pochodzą z pakiety R survival. Wybór tego zbioru danych był nieprzypadkowy. Dane pochodzą z prawdziwego badania oraz są relatywnie dobrze udokumentowane. <br>
# Cały zbiór składa się z 16 kolumn:
# -  id: id
# -  study: 1 for all patients
# -  rx: Treatment - Obs(ervation), Lev(amisole), Lev(amisole)+5-FU
# -  sex: 1=male
# -  age: in years
# -  obstruct: obstruction of colon by tumour
# -  perfor: perforation of colon
# -  adhere: adherence to nearby organs
# -  nodes: number of lymph nodes with detectable cancer
# -  time: days until event or censoring
# -  status: censoring status
# -  differ: differentiation of tumour (1=well, 2=moderate, 3=poor)
# -  extent: Extent of local spread (1=submucosa, 2=muscle, 3=serosa, 4=contiguous structures)
# -  surg: time from surgery to registration (0=short, 1=long)
# -  node4: more than 4 positive lymph nodes
# -  etype: event type: 1=recurrence,2=death
#

# %%
import pandas as pd
import matplotlib.pyplot as plt
import lifelines as lfl

# %%
colon_df = pd.read_csv('colon.csv')

# %%
colon_df.head()

# %%
colon_df.dropna(inplace=True)

# %%
colon_df = colon_df[colon_df['etype'] == 1]

# %%
len(colon_df)

# %%
T = colon_df['time']
E = colon_df['status']

# %%
# Overall KM curve
kmf = lfl.KaplanMeierFitter()
kmf.fit(durations=T, event_observed=E)
kmf.plot()

# %%
rx1 = colon_df['rx'] == 1
rx2 = colon_df['rx'] == 2
rx3 = colon_df['rx'] == 3

# %%
ax = plt.subplot(111)

kmf.fit(durations=T[rx1], event_observed=E[rx1], label='Observation')
kmf.plot(ax=ax)

kmf.fit(durations=T[rx2], event_observed=E[rx2], label='Lev(amisole)')
kmf.plot(ax=ax)

kmf.fit(durations=T[rx3], event_observed=E[rx3], label='Lev(amisole) + 5-FU')
kmf.plot(ax=ax)

# %%
m = colon_df['sex'] == 0

# %%
ax = plt.subplot(111)

kmf.fit(durations=T[m], event_observed=E[m], label='Male')
kmf.plot(ax=ax)

kmf.fit(durations=T[~m], event_observed=E[~m], label='Female')
kmf.plot(ax=ax)

# %%
# Cox PH model
cph = lfl.CoxPHFitter()
cph.fit(colon_df[['time', 'status', 'rx', 'sex']], duration_col='time', event_col='status')
cph.print_summary()

# %%
