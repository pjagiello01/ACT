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
# Projekt ma na celu zaproponowanie prostego schematu analizy przeżycia, w którym badać będziemy efektywność adiuwantowych chemioterapii w leczeniu raka jelita grubego na podstawie zbioru danych *colon* z pakietu survival dedykowanego dla języka R. 
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
# Jak już zostało wspomniane do projektu użyliśmy zbioru danych *colon* z pakietu survival. Zbiór stworzony został na podstawie jednej z pierwszych udanych prób leczenia raka jelita grubego poprzez adiuwantową chemioterapię. 
#  <br>
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

# %% [markdown]
# Aby nie komplikować zbytnio analizy, spośród wyżej wymieniobych wybraliśmy 6 zmiennych do naszej analizy, nie wliczając oczywistych zmiennych określających czas i status. Są to:
# - rx
# - sex
# - age
# - obstruct
# - adhere
# - differ
# <br> <br> Zmienne te zostały wybrane arbitralnie kierując się naszą niewielką wiedzą na temat raka jelita grubego oraz dokumentacją pakietu survival, która wskazuje na istniejące problemy ze zmienną node4.

# %% [markdown]
# Analizę zaczynamy od załadowania potrzebnych bibliotek

# %%
import pandas as pd
import matplotlib.pyplot as plt
import lifelines as lfl

# %% [markdown]
# ### Eksploracja danych

# %%
colon_df = pd.read_csv('colon.csv')

# %%
colon_df = colon_df[['time', 'status', 'rx', 'sex', 'age', 'obstruct', 'adhere', 'differ']]
colon_df.head()

# %% [markdown]
# Na zbiór danych składa się 888 rekordów

# %%
colon_df.dropna(inplace=True)
len(colon_df)

# %% [markdown]
# ### Analiza nieparametryczna
# W tej części przedstawimy standardowe krzywe przeżycia dla naszych zmiennych, pokażemy krzywe wyznaczające funkcje ryzyka wyznaczone estymatorem Nelsona-Aalena. Dla wszystkich zmiennych przeprowadzone zostaną również testy log-rank

# %%
T = colon_df['time']
E = colon_df['status']

# %%
# Overall KM curve
kmf = lfl.KaplanMeierFitter()
kmf.fit(durations=T, event_observed=E)
kmf.plot()

# %%
## Hazard function
naf = lfl.NelsonAalenFitter()
naf.fit(T,event_observed=E)

# %%
print(naf.cumulative_hazard_.head())

# %%
naf.plot_cumulative_hazard()

# %% [markdown]
# #### 1. Zmienna określająca sposób leczenia - *rx*

# %%
colon_df['rx'].value_counts()

# %%
## K-M curve
rx1 = colon_df['rx'] == 1
rx2 = colon_df['rx'] == 2
rx3 = colon_df['rx'] == 3


ax = plt.subplot(111)

kmf.fit(durations=T[rx1], event_observed=E[rx1], label='Observation')
kmf.plot(ax=ax)

kmf.fit(durations=T[rx2], event_observed=E[rx2], label='Lev(amisole)')
kmf.plot(ax=ax)

kmf.fit(durations=T[rx3], event_observed=E[rx3], label='Lev(amisole) + 5-FU')
kmf.plot(ax=ax)


# %%
## Hazard function
ax = plt.subplot(111)

naf.fit(durations=T[rx1], event_observed=E[rx1], label='Observation')
naf.plot(ax=ax)

naf.fit(durations=T[rx2], event_observed=E[rx2], label='Lev(amisole)')
naf.plot(ax=ax)

naf.fit(durations=T[rx3], event_observed=E[rx3], label='Lev(amisole) + 5-FU')
naf.plot(ax=ax)

# %%
## Log-Rank test

result = lfl.statistics.multivariate_logrank_test(colon_df['time'], colon_df['rx'], colon_df['status'])
result.print_summary()

# %% [markdown]
# #### 2. Zmienna określająca płeć - *sex*

# %%
colon_df['sex'].value_counts()

# %%
## K-M curve
m = colon_df['sex'] == 0


ax = plt.subplot(111)

kmf.fit(durations=T[m], event_observed=E[m], label='Male')
kmf.plot(ax=ax)

kmf.fit(durations=T[~m], event_observed=E[~m], label='Female')
kmf.plot(ax=ax)

# %%
## Hazard function

ax = plt.subplot(111)

naf.fit(durations=T[m], event_observed=E[m], label='Male')
naf.plot(ax=ax)

naf.fit(durations=T[~m], event_observed=E[~m], label='Female')
naf.plot(ax=ax)

# %%
## Log-Rank test
result = lfl.statistics.multivariate_logrank_test(colon_df['time'], colon_df['sex'], colon_df['status'])
result.print_summary()

# %% [markdown]
# #### 3. Zmienna określająca wiek - *age*

# %%
colon_df['age'].describe()

# %%
plt.hist(colon_df['age'])

# %%
bins = [0, 45, 65, float('inf')]
labels = ['young', 'mid', 'old']

colon_df['age_category'] = pd.cut(colon_df['age'], bins=bins, labels=labels)
colon_df.head()

# %%
## K-M curve
young = colon_df['age_category'] == 'young'
mid = colon_df['age_category'] == 'mid'
old = colon_df['age_category'] == 'old'


ax = plt.subplot(111)

kmf.fit(durations=T[young], event_observed=E[young], label='<45')
kmf.plot(ax=ax)

kmf.fit(durations=T[mid], event_observed=E[mid], label='45-65')
kmf.plot(ax=ax)

kmf.fit(durations=T[old], event_observed=E[old], label='65<')
kmf.plot(ax=ax)


# %%
## Hazard function
ax = plt.subplot(111)

naf.fit(durations=T[young], event_observed=E[young], label='Young')
naf.plot(ax=ax)

naf.fit(durations=T[mid], event_observed=E[mid], label='Mid')
naf.plot(ax=ax)

naf.fit(durations=T[old], event_observed=E[old], label='Old')
naf.plot(ax=ax)

# %%
## Log-Rank test
result = lfl.statistics.multivariate_logrank_test(colon_df['time'], colon_df['age_category'], colon_df['status'])
result.print_summary()

# %%

# %% [markdown]
# ### Model COX PH

# %%
# Cox PH model
cph = lfl.CoxPHFitter()
cph.fit(colon_df[['time', 'status', 'rx', 'sex', 'age']], duration_col='time', event_col='status')
cph.print_summary()

# %%
results = lfl.statistics.proportional_hazard_test(cph, colon_df[['time', 'status', 'rx', 'sex', 'age']], time_transform='rank')

# %%
results.print_summary()

# %%
