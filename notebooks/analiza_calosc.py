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
# https://cran.r-project.org/web/packages/survival/survival.pdf

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
import seaborn as sns
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# Ustawienie stylu wykresów
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.1)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True

# %% [markdown]
# ## Eksploracja danych
#
# Analizę rozpoczynamy od krótkiej eksploracji naszego zbioru danych. Jak już zostało wspomniane, do analizy wybraliśmy zmienne: 'time', 'status', 'rx', 'sex', 'age', 'obstruct', 'adhere', 'differ'. 
#
# Ponieważ chcemy badać wpływ terapi Lev(amisole) + 5-FU, wyrzucamy zmienną określającą terapię Lev(amisole), czyli 'rx' = 2

# %%
colon = pd.read_csv('../data/colon.csv')

# %%
colon = colon[['time', 'status', 'rx', 'sex', 'age', 'obstruct', 'adhere', 'differ']]

# %%
# Wyrzucamy Lev(amisole)
colon = colon[colon['rx'] != 2]

# %% [markdown]
# Na zbiór danych składa się 583 rekordów

# %%
len(colon)

# %%
colon.head()

# %%
colon.loc[colon['time'] == colon['time'].max(), 'status']

# %% [markdown]
# ## Analiza nieparametryczna
#
# W analizie nieparametrycznej wyciągamy wnioski bezpośrednio z naszych danych bez przyjmowania żadnych dodatkowych założeń. Jej celem jest uzyskanie obserwowalnej krzywej przeżycia (survival curve) oraz obserwowalnej funkcji hazardu. Ponadto dla zmiennych kategorialnych możemy wizualnie sprawdzić wpływ zmiennej na krzywą przeżycia. 
#
# Krzywą przeżycia wyznaczymy stosując estymator Kaplana-Meiera, natomiast funkcję hazardu wyznaczymy przy pomocy estymatora Nelsona-Aalena.
#
# Kolejno przejdziemy przez następujące punkty:
# - analiza korelacji zmiennych
# - estymacja krzywej przeżycia
# - estymacja funkcji hazardu
# - wizualna ocena wpływu zmiennych na krzywą przeżycia oraz testy log-rank

# %% [markdown]
# ### Analiza korelacji

# %%
# Tworzymy dataframe korelacji zmiennych
correlations = colon.corr()['time']
correlations_df = pd.DataFrame(correlations)
correlations_df.drop(['status', 'time'], inplace=True)
correlations_df.reset_index(inplace=True)
correlations_df.columns = ['zmienna', 'korelacja z czasem przeżycia']

# %%
# Wizualizacja
plt.figure(figsize=(8, 5))
sns.barplot(
    data=correlations_df,    
    y='zmienna',
    x='korelacja z czasem przeżycia',
    palette='coolwarm_r', 
    orient='h'
)

# %% [markdown]
# Z powyższego wykresu wynika, że najsilniej skorelowaną z czasem przeżycia zmienną jest zmienna rx oznaczająca terapię. Zmienna ta jest pozytywnie skorelowana, co oznacza, że jej większa wartość idzie w parze z wydłużonym czasem przeżycia. Należy zatem zaznaczyć, że rx = 0 oznacza zwykłą obserwację pacjenta, natomiast rx = 3 oznacza terapię Lev(amisole) + 5-FU.
#
# Obserwujemy również dosyć silną ujemną korelację zmiennej differ z czasem przeżycia. Zmienna differ określa poziom różnicy komórek rakowych ze zdrowymi komórkami, im wyższy poziom, tym gorzej.

# %% [markdown]
# ### Krzywa przeżycia

# %% [markdown]
# Określa prawdopodobieństwo, że zmienna losowa $T$ (w naszym przypadku czas przeżycia pacjenta z rakiem jelita grubego) przekroczy $t$

# %%
T = colon['time']
E = colon['status']

# %%
# Krzywa przeżycia - Kaplan-Meier
from lifelines.plotting import add_at_risk_counts

plt.figure(figsize=(12, 6))
ax = plt.subplot(111)

kmf = lfl.KaplanMeierFitter()
kmf.fit(durations=T, event_observed=E, alpha=0.05)
kmf.plot(ax=ax, ci_show=True, ci_alpha=0.3)
add_at_risk_counts(kmf, ax=ax)

plt.title('Krzywa przeżycia Kaplana-Meiera z 95% przedziałem ufności')
plt.xlabel('Czas (dni)')
plt.ylabel('Prawdopodobieństwo przeżycia')
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# Z powyższej krzywej przeżycia można wywnioskować, żę największe ryryko śmierci jest na samym początku. Prawdopodobnie są to najcięższe przypadki nowotworu jelita grubego. Później jednak krzywa się wygładza i tempo przyrostu obserwacji oznaczanych jako failure zdecydowanie maleje. Jest sporo ocenzurowanych pacjentów, czyli takich, dla których nie mamy informacji na temat wystąpienia zdarzenia (w naszym przypadku po prostu ocenzurowani pacjenci nie umarli w trakcie badania). Z krzywej wynika, że ponad połowa chorzych na raka jelita grubego przeżywa ponad 9 lat.

# %%
### Tabela przeżycia Kaplana-Meiera
print("\nTabela przeżycia Kaplana-Meiera")

### Tworzenie tabeli z wynikami
kmf_table = pd.DataFrame(kmf.survival_function_)
kmf_table.columns = ['Przeżycie']
kmf_table['Dolny przedz. ufności'] = kmf.confidence_interval_['KM_estimate_lower_0.95']
kmf_table['Górny przedz. ufności'] = kmf.confidence_interval_['KM_estimate_upper_0.95']
kmf_table['Skumulowana funkcja hazardu'] = -(np.log(kmf.survival_function_.values))
kmf_table['Skumulowana funkcja hazardu'][0] = 0 ### korekta wizualna
kmf_table['Zdarzenia'] = kmf.event_table['observed']
kmf_table['Cenzorowane'] = kmf.event_table['censored']
kmf_table['Narażeni na ryzyko'] = kmf.event_table['at_risk']


print("Fragment tabeli przeżycia (pierwsze 10 wierszy):")
display(kmf_table.head(10))
print("\nFragment tabeli przeżycia (ostatnie 10 wierszy):")
display(kmf_table.tail(10))


# %% [markdown]
# ### Funckja hazardu

# %%
## funkcja hazardu
naf = lfl.NelsonAalenFitter()
naf.fit(T,event_observed=E)

# %%
plt.figure(figsize=(12, 6))
naf.plot_cumulative_hazard()
plt.title('Krzywa hazardu Nelsona-Aalena')
plt.xlabel('Czas przeżycia')
plt.ylabel('hazard')
plt.grid(alpha = 0.3)
plt.show()

# %% [markdown]
# Skumulowane ryzyko śmierci zwiększa się z czasem, co jest całkowicie zasadne. Początkowo ten wzrost jest wysoki, co widzieliśmy również na krzywej przeżycia. Później tempo wzrostu ryzyka maleje.

# %% [markdown]
# ### Zmienne

# %% [markdown]
# #### 1. Zmienna określająca sposób leczenia - *rx*
#
# 1 - obserwacja
#
# 3 - terapia lewamizolem +5-FU

# %%
colon['rx'].value_counts()

# %%
## K-M curve
rx1 = colon['rx'] == 1
rx3 = colon['rx'] == 3

plt.figure(figsize=(12, 6))
ax = plt.subplot(111)
kmf.fit(durations=T[rx1], event_observed=E[rx1], label='Observation', alpha=0.05)
kmf.plot(ax=ax)
kmf.fit(durations=T[rx3], event_observed=E[rx3], label='Lev(amisole) + 5-FU', alpha=0.05)
kmf.plot(ax=ax)
plt.grid(alpha = 0.3)

# %% [markdown]
# Krzywe narysowane z 95% przedziałem ufności. Krzywa dla pacjentów, którzy leczeni są terapią Lev(amisole) + 5-FU jest istotnie powyżej krzywej dla pacjentów tylko obserwowanych. Możemy wnioskować, że terapia zwiększa czas przeżycia.

# %%
plt.figure(figsize=(12, 6))

ax = plt.subplot(111)

naf.fit(durations=T[rx1], event_observed=E[rx1], label='Observation', alpha=0.05)
naf.plot(ax=ax)


naf.fit(durations=T[rx3], event_observed=E[rx3], label='Lev(amisole) + 5-FU', alpha=0.05)
naf.plot(ax=ax)

plt.grid(alpha = 0.3)

# %%
## Log-Rank test

result = lfl.statistics.multivariate_logrank_test(colon['time'], colon['rx'], colon['status'])
result.print_summary()

# %% [markdown]
# Wyniki testu log-rank pozwalają nam odrzucić hipotezę zerową zakładającą brak różnicy rozkładów czasu przeżycia dla badanych kategorii. Możemy podejrzeważ, że zmienna rx okaże się istotna w modelach parametrycznych i semiparametrycznych.

# %% [markdown]
# #### 2. Zmienna określająca płeć - *sex*
#
# 0 - mężczyzna
#
# 1 - kobieta

# %%
colon['sex'].value_counts()

# %%
## Krzywa K-M
m = colon['sex'] == 0

plt.figure(figsize=(12, 6))
ax = plt.subplot(111)

kmf.fit(durations=T[m], event_observed=E[m], label='Male', alpha=0.05)
kmf.plot(ax=ax)

kmf.fit(durations=T[~m], event_observed=E[~m], label='Female', alpha=0.05)
kmf.plot(ax=ax)

plt.grid(alpha = 0.3)

# %%
## Funkcja hazardu

plt.figure(figsize=(12, 6))
ax = plt.subplot(111)

naf.fit(durations=T[m], event_observed=E[m], label='Male', alpha=0.05)
naf.plot(ax=ax)

naf.fit(durations=T[~m], event_observed=E[~m], label='Female', alpha=0.05)
naf.plot(ax=ax)

plt.grid(alpha = 0.3)

# %%
## Log-Rank test
result = lfl.statistics.multivariate_logrank_test(colon['time'], colon['sex'], colon['status'])
result.print_summary()

# %% [markdown]
# Przy poziomie istotności 5% nie możemy odrzucić hipotezy zerowej.

# %% [markdown]
# #### 3. Zmienna obstruct
#
# Określa, czy guz blokuje drożność jelita. Jeśli tak, to zmienna przyjmuje wartość 1, przeciwnie 0

# %%
colon['obstruct'].value_counts()

# %%
## Krzywa K-M
o = colon['obstruct'] == 0

plt.figure(figsize=(12, 6))
ax = plt.subplot(111)

kmf.fit(durations=T[o], event_observed=E[o], label='No obstruct', alpha=0.05)
kmf.plot(ax=ax)

kmf.fit(durations=T[~o], event_observed=E[~o], label='Obstruct', alpha = 0.05)
kmf.plot(ax=ax)

plt.grid(alpha = 0.3)

# %%
## Funkcja hazardu

plt.figure(figsize=(12, 6))
ax = plt.subplot(111)

naf.fit(durations=T[o], event_observed=E[o], label='No obstruct', alpha=0.05)
naf.plot(ax=ax)

naf.fit(durations=T[~o], event_observed=E[~o], label='Obstruct', alpha=0.05)
naf.plot(ax=ax)

plt.grid(alpha = 0.3)

# %%
## Log-Rank test
result = lfl.statistics.multivariate_logrank_test(colon['time'], colon['obstruct'], colon['status'])
result.print_summary()

# %% [markdown]
# Podobnie jak w przypadku płci, nie ma podstaw do odrzucenia hipotezy zerowej zakładającej identyczność krzywych przeżycia dla wszystkich kategorii, przy poziomie istotności 95%. Interesujące jest przycięcie dla kategorii No obstruct na wykresach wskazujące na to, że ostatni pacjent z tej kategorii został ocenzurowany wcześniej niż ostatni pacjent z kategorii Obstruct. Nie mamy szczegółowych informacji na temat oryginalnego badania, więc możemy jedynie snuć domysły na temat przyczyny takiego stanu rzeczy.

# %% [markdown]
# #### 4. Zmienna adhere

# %%
colon['adhere'].value_counts()

# %%
## Krzywa K-M
a = colon['adhere'] == 0

plt.figure(figsize=(12, 6))
ax = plt.subplot(111)

kmf.fit(durations=T[a], event_observed=E[a], label='No adhere', alpha = 0.05)
kmf.plot(ax=ax)

kmf.fit(durations=T[~a], event_observed=E[~a], label='adhere', alpha = 0.05)
kmf.plot(ax=ax)

plt.grid(alpha = 0.3)

# %%
## Funkcja hazardu

plt.figure(figsize=(12, 6))
ax = plt.subplot(111)

naf.fit(durations=T[a], event_observed=E[a], label='No adhere', alpha = 0.05)
naf.plot(ax=ax)

naf.fit(durations=T[~a], event_observed=E[~a], label='Adheree', alpha = 0.05)
naf.plot(ax=ax)

plt.grid(alpha = 0.3)

# %%
## Log-Rank test
result = lfl.statistics.multivariate_logrank_test(colon['time'], colon['adhere'], colon['status'])
result.print_summary()

# %% [markdown]
# Odrzucenie hipotezy zerowej zakładającej identyczność krzywych przeżycia dla wszystkich kategorii, przy poziomie istotności 95%. Patrząc na wykres jest to dosyć interesujący wynik testu Log-Rank. Widoczny jest niewielki interwał na którym przedziały ufności się nie pokrywają. Krzywa przeżycia dla kategorii *adhere* leży wyraźnie niżej, ale ma ona również zdecydowanie szerszy przedział ufności. Oznacza to, że jesteśmy bardziej pewni kształtu krzywej przeżycia dla pacjentów z guzem nieadherentnym. 
#

# %% [markdown]
# #### 5. Zmienna differ
#
#

# %%
colon['differ'].value_counts()

# %%
## Krzywa K-M
d1 = colon['differ'] == 1
d2 = colon['differ'] == 2
d3 = colon['differ'] == 3

plt.figure(figsize=(12, 6))
ax = plt.subplot(111)

kmf.fit(durations=T[d1], event_observed=E[d1], label='differ = 1')
kmf.plot(ax=ax)

kmf.fit(durations=T[d2], event_observed=E[d2], label='differ = 2')
kmf.plot(ax=ax)

kmf.fit(durations=T[d3], event_observed=E[d3], label='differ = 3')
kmf.plot(ax=ax)

plt.grid(alpha = 0.3)

# %%
## Log-Rank test
result = lfl.statistics.multivariate_logrank_test(colon['time'], colon['adhere'], colon['status'])
result.print_summary()

# %% [markdown]
# Odrzucenie hipotezy zerowej zakładającej identyczność krzywych przeżycia dla wszystkich kategorii, przy poziomie istotności 95%. Dosyć wyraźnie widać to również na wykresie. Sam wykres poniekąd potwierdza nasze wcześniejsze przypuszczenia (przynajmniej wizualnie), że początkowe wysokie nachylenie krzywej przeżycia jest wynikiem sporej ilości śmierci pacjentów z nowotworami bardziej agresywnymi. Tutaj te nowotwory pośrednio określa zmienna differ = 3. Dokładnie oznacza ona najsłabsze zróżnicowanie komórek rakowych w rozumieniu takim, że nie są one podobne do zdrowych komórek. To najczęściej charakteryzuje bardziej agresywne nowotwory.

# %% [markdown]
# ## Analiza parametryczna

# %% [markdown]
# Analiza parametryczna w analizie przeżycia polega na założeniu określonego rozkładu prawdopodobieństwa czasu przeżycia, takiego jak rozkład wykładniczy, Weibulla, log-normalny czy gamma. W przeciwieństwie do metod nieparametrycznych, analiza parametryczna umożliwia także prognozowanie przeżycia poza obserwowanym zakresem danych. Jednak ważnym ograniczeniem tych metod jest konieczność poprawnego doboru rozkładu, gdyż błędne założenia mogą prowadzić do niewiarygodnych wyników.
#
# W naszej analizie uwzględniamy 3 rozkłady:
# - rozkład Weibulla
# - rozkład log-normalny
# - rozkład log-logistyczny
#
# Wszystkie zastosowane przez nas modele są modelami AFT
#
# Zanim jednak przejdziemy do analizy parametrycznej należy odpowiednio przygotować dane. Mamy zmienne, które mają więcej niż 2 kategorie. Te zmienne muszą zostać przedefiniowane jako zmienne zero-jedynkowe

# %%
colon = pd.get_dummies(colon, columns = ['differ'], drop_first=True)
colon.head()

# %% [markdown]
# ### 1. Wybór modelu

# %%
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter

wb_aft, ln_aft, ll_aft = WeibullAFTFitter(), LogNormalAFTFitter(), LogLogisticAFTFitter()
wb_aft.fit(colon, duration_col = 'time', event_col = 'status')
ln_aft.fit(colon, duration_col = 'time', event_col = 'status')
ll_aft.fit(colon, duration_col = 'time', event_col = 'status')


# %% [markdown]
# ### Porównanie modeli

# %% [markdown]
# Porównanie prognozowanych krzywych przeżycia dla pierwszej obserwacji, czyli tej, dla tej z najwcześniejszym failure

# %%
colon.iloc[0:1]

# %%
plt.figure(figsize=(12, 6))
plt.plot(wb_aft.predict_survival_function(colon.iloc[0:1]), label = 'Weibull')
plt.plot(ln_aft.predict_survival_function(colon.iloc[0:1]), label = 'Log-Normal')
plt.plot(ll_aft.predict_survival_function(colon.iloc[0:1]), label = 'Log-Logistic')
plt.grid(alpha = 0.3)
plt.title('Porównanie prognoz dla wybranego pacjenta')
plt.legend()
plt.show()

# %% [markdown]
# Estymowane wykresy pokazują, że pacjent względem całej populacji nie rokował dobrze. Jest to spodziewane, jako, że był to pacjent z najwcześniejszym failure.

# %% [markdown]
# Porównanie wartości funkcji największej wiarygodności

# %%
print(f"Weibull LL: {wb_aft.log_likelihood_}")
print(f"Log-Normal LL: {ln_aft.log_likelihood_}")
print(f"Log-Logistic LL: {ll_aft.log_likelihood_}")

# %%
models = {
    'Weibull': wb_aft,
    'LogNormal': ln_aft,
    'LogLogistic': ll_aft
}

#obliczamy AIC modeli
aic_values = {}
c_values = {}
for name, model in models.items():
    aic_values[name] = model.AIC_
    c_values[name] = model.concordance_index_

for name, model in models.items():
    print(f"\n{name} Podsumowanko:")
    print(f"AIC: {model.AIC_}")
    print(f"Concordance_index: {model.concordance_index_}")

#najlepszy model
best_model_name = min(aic_values, key=aic_values.get)
best_model = models[best_model_name]
print(f"\nNajlepszy model na podstawie AIC: {best_model_name} (AIC={aic_values[best_model_name]})")
best_model_name = max(c_values, key=c_values.get)
best_model = models[best_model_name]
print(f"\nNajlepszy model na podstawie Concordance index: {best_model_name} (Concordance index={c_values[best_model_name]})")


# %%
c_values

# %%
models = {
    'Weibull': 'Weibull',
    'LogNormal': 'LogNormal',
    'LogLogistic': 'LogLogistic'
}
aicdf=pd.DataFrame({
    "Model": models,
    "AIC": aic_values,
})
cdf=pd.DataFrame({
    "Model": models,
    "Concordance index": c_values,
})
# Wykres AIC
plt.figure(figsize=(8, 5))
plt.bar(aicdf['Model'], aicdf['AIC'], color='skyblue')
plt.title('Porównanie modeli AFT - AIC')
plt.ylabel('AIC')
plt.xlabel('Model')
plt.grid(axis='y')
plt.ylim(4500, max(aicdf['AIC']) * 1.01)  
plt.tight_layout()
plt.show()

# Wykres Concordance Index
plt.figure(figsize=(8, 5))
plt.bar(cdf['Model'], cdf['Concordance index'], color='lightgreen')
plt.title('Porównanie modeli AFT - Concordance index')
plt.ylabel('Concordance index')
plt.xlabel('Model')
plt.grid(axis='y')
plt.ylim(0.6, 0.6150)  
plt.tight_layout()
plt.show()

# %% [markdown]
# Najlepszy model to model z rozkładem lognormalnym - najwyższy log likelihhod, najniższy AIC oraz najwyższy concordance. Przyjżymy się teraz nieco dokładniej temu modelowi.

# %%
ln_aft = LogNormalAFTFitter()
ln_aft.fit(colon, duration_col='time', event_col='status')
ln_aft.print_summary()

# %%
#wykres log(accelerated failure)
fig, ax = plt.subplots(figsize=(10, 6))
ln_aft.plot(ax=ax)
plt.title("Współczynniki modelu Log-Normal AFT")
plt.grid(True)
plt.show()

# %% [markdown]
# statystycznie istotne zmienne: differ, rx, age_category_mid, age_catefory_old, differ_3, sex. Ale zostawiamy również zmienną obstruct

# %%
ln_aft = LogNormalAFTFitter()
ln_aft.fit(colon[['time', 'status', 'differ_3', 'rx', 'sex', 'obstruct']], duration_col='time', event_col='status')
ln_aft.print_summary()

# %%
#wykres log(accelerated failure)
fig, ax = plt.subplots(figsize=(10, 6))
ln_aft.plot(ax=ax)
plt.title("Współczynniki modelu Log-Normal AFT")
plt.grid(True)
plt.show()

# %%
#mediana i średnia
print(f"Mediana: {ln_aft.median_survival_time_}")
print(f"Średnia: {ln_aft.mean_survival_time_}")

# %%
#funkcja przeżycia dla 1 obserwacji
surv_func = ln_aft.predict_survival_function(colon.iloc[0:1])
fig, ax = plt.subplots(figsize=(10, 6))
surv_func.plot(ax=ax, title="Funkcja przeżycia dla 1 obserwacji - LogNormal model")
ax.set_xlabel("Time (days)")
ax.set_ylabel("Survival probability")
ax.grid(True)
plt.show()

# %%
#wykresy partial effects on outcome
fig, ax = plt.subplots(figsize=(10, 6))
ln_aft.plot_partial_effects_on_outcome('sex', (0, 1), cmap='coolwarm', ax=ax)
ax.set_title('Wpływ płci na czas przeżycia')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 6))
ln_aft.plot_partial_effects_on_outcome('rx', (1,3), cmap='coolwarm', ax=ax)
ax.set_title('Wpływ leczenia na czas przeżycia')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 6))
ln_aft.plot_partial_effects_on_outcome('differ_3',(0,1), cmap='coolwarm', ax=ax)
ax.set_title('Wpływ zróżnicowania histologicznego na czas przeżycia')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 6))
ln_aft.plot_partial_effects_on_outcome('obstruct',(0,1), cmap='coolwarm', ax=ax)
ax.set_title('Wpływ zatykania jelita przez guza na czas przeżycia')
plt.show()

# %%
# Funkcja hazardu 
plt.figure(figsize=(10, 6))
plt.plot(ln_aft.predict_cumulative_hazard(colon.iloc[0:1]))
plt.title('Funkcja hazardu – LogNormal model')
plt.xlabel('Time')
plt.ylabel('Hazard')
plt.grid(True)
plt.legend()
plt.show()

# %%
# Po dopasowaniu modelu AFT, uzyskujemy macierz kowariancji:
cov_matrix = ln_aft.variance_matrix_

# Konwersja macierzy kowariancji do DataFrame (czytelniejszy format)
cov_matrix_df = pd.DataFrame(cov_matrix)

# %%
# Macierz kowariancji współczynników (jak COVB w SAS)
print("\nMacierz kowariancji (COVB):")
print(cov_matrix_df)

# %% [markdown]
# ## Analiza semiparametryczna - model parametrycznych hazardów Coxa

# %%
# Cox PH model
cph = lfl.CoxPHFitter()
cph.fit(colon[['time', 'status', 'rx']], duration_col='time', event_col='status')
cph.print_summary()

# %%
# Cox PH model
cph = lfl.CoxPHFitter()
cph.fit(colon, duration_col='time', event_col='status')
cph.print_summary()

# %%
cph = lfl.CoxPHFitter()
cph.fit(colon[['time', 'status', 'sex', 'rx', 'differ_3', 'obstruct']], duration_col='time', event_col='status')
cph.print_summary()

# %%
cph.check_assumptions(colon[['time', 'status', 'sex', 'rx', 'differ_3', 'obstruct']], p_value_threshold=0.05, show_plots= True)

# %% [markdown]
# Założenie proporcjonalnych hazardów nie jest spełnione dla zmiennej differ_3. Aby skorygować model proporcjonalnych hazardów tak, aby uwzględniał nieproporcjonalne bazowe funkcje hazardu dla zmiennej differ_3 możemy spróbować stratyfikacji. Stratyfikacja generalnie polega na podziale zmiennej na podgrupy i wyestymowaniu osobnej bazowej funkcji hazardu dla każdej z nich. Zmienna differ_3 oczywiście ma tylko 2 grupy, także dla tych 2 grup estymujemy bazową funkcję hazardu.  

# %%
print(colon[['differ_3']].info())
print(colon['differ_3'].unique())
print(colon['differ_3'].isna().sum())

# %% [markdown]
# Musimy odpowiedno przygotować zmienną, tak aby poprawnie przekazać ją do funckji jako strat

# %%
colon_strat = colon.copy()
colon_strat['differ_3'] = colon_strat['differ_3'].astype(str)

# %%
cph = lfl.CoxPHFitter()

cph.fit(
    colon_strat[['time', 'status', 'sex', 'rx', 'obstruct', 'differ_3']], 
    duration_col='time', 
    event_col='status', 
    strata=['differ_3']
)

cph.print_summary()

# %% [markdown]
# szacowania parametrów są podobne jak w poprzednich modelach Coxa. Uwagę powinien zwrócić brak parametru dla zmiennej differ_3. Jest to spodziewane, mamy wyestymowane 2 nieproporcjonalne bazowe funkcje hazardu, więc nie mamy współczynnika określającego proporcję hazardów. 
# Natomiast najbardziej interesuje nas, czy założenia modelu są spełnione. Sprawdzamy to za pomocą check_assumptions w pakiecie lifelines.
#

# %%
cph.check_assumptions(colon_strat[['time', 'status', 'sex', 'rx', 'differ_3', 'obstruct']], p_value_threshold=0.05, show_plots= False)

# %% [markdown]
# Dodajemy również model ze zmienną differ_3 zależną liniowo od czasu. Żeby to zrobić trzeba najpierw odpowiednio przygotować dane

# %%
colon_tv = colon.copy()
colon_tv['differ_3*time'] = colon_tv['differ_3']*colon_tv['time']

# %%
from lifelines.utils import to_episodic_format

colon_tv = to_episodic_format(colon[['time', 'status', 'rx', 'sex', 'obstruct', 'differ_3']], duration_col='time', event_col='status', time_gaps=100.)
colon_tv.head(25)

# %%
colon_tv['differ_3*time'] = colon_tv['differ_3']*colon_tv['stop']

# %%
colon_tv['stop']

# %%
from lifelines import CoxTimeVaryingFitter

ctv = CoxTimeVaryingFitter()

ctv.fit(colon_tv,
        id_col='id',
        event_col='status',
        start_col='start',
        stop_col='stop')

# %%
ctv.print_summary()
