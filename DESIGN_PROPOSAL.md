# Design proposal

## Topic

| The question | *Do deletion with zero fill, leak-free imputation, retraining, infidelity and sensitivity agree with each other and with a planted ground truth on audio detectors, and does the ranking of methods change with the metric?* |
| --- | --- |
| What is known | Remove-and-retrain (2018) and remove-and-debias (2022) define the reference and the leak-free shortcut. Infidelity and sensitivity (2019) define faithfulness over a perturbation distribution. Planted-feature benchmarks (2021, 2024) give ground truth in vision. Metrics fail reliability tests (2020, 2023). In audio, a 2026 framework uses band deletion, a 2026 study scores Integrated Gradients against synthetic event timestamps, and a 2024 Grad-CAM study found that partial-spoof detectors attend to the splice transitions, which is a warning about ground truth. No study scores the metrics against a planted ground truth on audio detectors. |
| Terms used | • **<u>deletion curve</u>** - the output as a function of how much input was removed in the map's order. The fill value (zeros, mean, shaped noise) is a choice.<br>**• <u>leak-free imputation</u>** - filling removed regions from their neighbours plus noise so the hole's shape does not tell the model the class.<br>**• <u>planted feature</u>** - an artifact added to real clips of one class so that a model trained on them is known to use it. |
| First steps | 1. On PartialSpoof with one detector, compute four methods' maps for 100 clips and the share of each map's mass in the synthetic segment.<br>2. Compute deletion curves with zero fill and with noise fill and compare the method rankings.<br>3. Write down the refuting result. |
| Plan | 1. Build ground truth A from partially spoofed speech with the given boundaries, and ground truth B from four planted artifacts at three strengths in real clips, with a detector trained to use each, a control detector trained on unplanted data, and real-into-real splices as the boundary control.<br>2. Measure the recovery of five methods on A and B with a random-map control.<br>3. Run every metric on the same maps: deletion and insertion with three fills, leak-free imputation, head-only retraining, infidelity, sensitivity.<br>4. Compute the rank agreement of each metric with the ground truth and among metrics, and run the reliability tests on the metrics themselves.<br>5. Text variant: plant a lexical cue in human text, use paraphrase as the transformation, Integrated Gradients and occlusion over tokens, the same metrics. Write the recommendation and release both ground-truth sets. |
| What you deliver | You deliver two ground-truth sets with their controls, the recovery scores of five methods, every metric's ranking of them, the rank-agreement table that says which metrics can be trusted, the reliability verdict on the metrics, and a recommendation of what to report. If all metrics agree, you have shown that the vision concern does not carry to audio detectors, and the benchmark remains. |
| What would refute it | All metrics rank the methods identically and agree with the ground truth within intervals on both sets at every strength. |
| Where the paper could go | ICASSP. Interspeech. A datasets and benchmarks track. A workshop on explainable machine learning. |
| What you need | PartialSpoof and ASVspoof 2019. One small detector that retrains in a day. Captum, Zennit, Quantus. One GPU, with a few GPU-days for retraining. The text variant needs a public text detector with continuous scores and a paraphraser. |
| Watch out for | A planted artifact can be too easy or ignored. The three strengths and the control detector show which. A detector may use the splice boundary rather than the segment. The real-into-real control measures that. |
| Read first | • https://arxiv.org/abs/1806.10758<br>• https://arxiv.org/abs/2202.00449<br>• https://arxiv.org/abs/1901.09392<br>• https://arxiv.org/abs/2104.14403<br>• https://arxiv.org/abs/2302.07265<br>• https://arxiv.org/abs/2601.19017<br>• https://arxiv.org/abs/2605.23293<br>• https://arxiv.org/abs/2104.02518<br>• https://arxiv.org/abs/2406.02483<br>• https://arxiv.org/abs/2301.11305<br>• https://arxiv.org/abs/2606.14466 |


## Calendar

| Tydzień | Deadline | Opis |
| --- | --- | --- |
| #1 (22.10 - 28.10) | **28.10 - design proposal** | Design proposal, w tym przegląd załączonej literatury oraz plan i założenia projektu. |
| #2 (29.10 - 04.11) | | Wytrenowanie/uruchomienie bazowego detektora `ResNet-18` na 100 próbkach `PartialSpoof` oraz wyznaczenie map atrybucji dla 4 metod (`Integrated Gradients`, `SmoothGrad-Squared`, `Grad-CAM`, `Occlusion`) i policzenie udziału wyjaśniania w segmencie syntetycznym. |
| #3 (05.11 - 11.11) | **06.11 - prototyp deadline na spotkanie** | Wyznaczenie krzywych *deletion* dla wypełnienia zerami (*zero fill*) oraz szumem (*noise fill*), porównanie wstępnych rankingów metod. Prezentacja prototypu. |
| #4 (12.11 - 18.11) | | Przygotowanie pełnego **Ground Truth A** na podstawie segmentów `PartialSpoof` oraz **Ground Truth B** na bazie `ASVspoof 2019 LA` (dodanie 4 sztucznych artefaktów z różną siłą + zbiór kontrolny bez artefaktów). |
| #5 (19.11 - 25.11) | | Nauka na kolokwium. |
| #6 (26.11 - 02.12) | | Trenowanie detektorów na danych z artefaktami oraz modelu kontrolnego, sprawdzenie, czy modele rzeczywiście wykorzystują planted features. |
| #7 (03.12 - 09.12) | | Wyznaczenie map atrybucji dla pełnego zestawu **5 metod** (`IG`, `SG-SQ`, `Grad-CAM`, `LRP`, `Occlusion`) + losowej mapy kontrolnej (`Random`) na zbiorach GT A i GT B. |
| #8 (10.12 - 16.12) | | Uruchomienie metryk niewymagających douczania na wyznaczonych mapach: *Deletion* (3 rodzaje: zero, średnia, szum), *Insertion* (3 rodzaje), *Leak-free imputation (ROAD)*, *Infidelity* oraz *Max-Sensitivity*. |
| #9 (17.12 - 23.12) | | Przeprowadzenie eksperymentów **Head-only Retraining (ROAR-Head)** dla kolejnych progów degradacji na zbiorach GT A i GT B. |
| #10 (24.12 - 30.12) | | Obliczenie tabel korelacji między metrykami i GT oraz uruchomienie testów niezawodności samych metryk (MetaQuantus).. |
| #11 (31.12 - 06.01) | | **Text Variant:** Wstrzyknięcie fragmentu do tekstów ludzkich, użycie parafrazy jako transformacji, wyznaczenie map `IG` i `Occlusion` po tokenach dla detektora `RoBERTa` i ewaluacja tych samych metryk. |
| #12 (07.01 - 13.01) | | Spisanie finalnej wersji raportu i poprawki. |
| #13 (14.01) | **14.01 - oddanie projektu** |  |


## Related work

* **Hooker et al. (2019) - A Benchmark for Interpretability Methods in Deep Neural Networks:**
Wykazali, że zwykłe usuwanie najważniejszych pikseli zmienia rozkład danych, przez co spadek skuteczności modelu wynika z pojawienia się sztucznych plam, a nie z utraty istotnej informacji. Zaproponowali trenowanie modelu od nowa na zamaskowanych danych, udowadniając, że większość klasycznych metod atrybucji działa na równi z losowym zgadywaniem, a skuteczne są jedynie metody SmoothGrad-Squared oraz VarGrad.

* **Yeh et al. (2019) - On the (In)fidelity and Sensitivity of Explanations:**
Wprowadzili dwie miary oceny wyjaśnień niewymagające douczania sieci: Infidelity, badającą jak dobrze mapa przewiduje zmianę predykcji przy dużych zaburzeniach wejścia, oraz Sensitivity, mierzącą stabilność samej mapy przy drobnym szumie. Udowodnili, że większość znanych metod XAI to szczególne przypadki optymalizacji Infidelity, a wygładzanie map metodą SmoothGrad pozwala jednocześnie zwiększyć ich stabilność i wierność.

* **Rong et al. (2022) - A Consistent and Efficient Evaluation Strategy for Attribution Methods:**
Udowodnili, że zastępowanie usuwanych pikseli stałą wartością (nawet podczas retrainingu w ROAR) powoduje wyciek informacji o klasie przez sam kształt wyciętej dziury, co prowadzi do sprzecznych rankingów metod. Zaproponowali wypełnianie usuniętych miejsc średnią z sąsiednich pikseli z dodatkiem szumu, co eliminuje wyciek maski i daje spójne wyniki bez konieczności ponownego trenowania modelu.

* **Zhang et al. (2021) - An Initial Investigation for Detecting Partially Spoofed Audio:**
Wprowadzili zbiór PartialSpoof, zbudowany na podstawie ASVspoof 2019 LA przez łączenie fragmentów mowy prawdziwej i syntetycznej (wybrane fragmenty mowy prawdziwej zastąpione syntetyczną), z etykietami całych nagrań oraz poszczególnych segmentów. Wykazali, że detektor trenowany na prawdziwych i całkowicie syntetycznych nagraniach znacznie gorzej wykrywa częściowe zmiany, osiągając EER (Equal Error Rate) 15.96% na PartialSpoof wobec 2.65% na ASVspoof 2019 LA. Trenowanie na częściowych zmianach poprawia generalizację, ale dokładna lokalizacja syntetycznych fragmentów pozostaje trudniejsza niż klasyfikacja całego nagrania.

* **Liu et al. (2024) - How Do Neural Spoofing Countermeasures Detect Partially Spoofed Audio?:**
Autorzy sprawdzali, czy detektor rzeczywiście rozpoznaje charakterystyczne właściwości wygenerowanego głosu, czy może wykorzystuje łatwiejsze wskazówki. Zastosowali Grad-CAM oraz zaproponowaną miarę Relative Contribution Quantification (RCQ), porównującą średnią atrybucję w różnych typach segmentów nagrania. Wykazali, że badane detektory trenowane na PartialSpoof skupiają się przede wszystkim na artefaktach powstałych podczas łączenia fragmentów (na przejściach), podczas gdy model trenowany na całkowicie syntetycznych nagraniach silniej reaguje na samą mowę. 

* **Dumpis, Virtanen (2026) - Evaluating the Temporal Detection Capability of Integrated Gradients Applied on Sound Classifier:**
Autorzy sprawdzają, czy mapa atrybucji modelu klasyfikującego całe nagranie może wskazywać, kiedy dane zdarzenie występuje, nawet jeśli model nigdy nie otrzymał dokładnych etykiet czasowych podczas treningu. Oceniali to bazując na Integrated Gradients dla klasyfikatora CNN14 na syntetycznych nagraniach nakładających się zdarzeń dźwiękowych z dokładnymi znacznikami czasowymi. Uzyskali wyniki przewyższające mapy losowe i bazę opartą na energii sygnału (średnie IoU 0.39, F1 0.52 i Pointing Game 82,6%), choć ustępujące modelom przewidującym aktywność zdarzeń w kolejnych ramkach. Analiza IG obejmowała wyłącznie klasy z przewidywanym prawdopodobieństwem powyżej 0.5, więc wyniki lokalizacji są warunkowe względem wykrycia klasy. Pokazali też znaczący wpływ progu atrybucji na ocenę. 


## Datasets

1. **`PartialSpoof Database`** Zbiór zawierający nagrania prawdziwe oraz częściowo zmanipulowane, w których fragmenty mowy ludzkiej zastąpiono mową syntetyczną z dokładnymi adnotacjami czasowymi granic segmentów.
2. **`ASVspoof 2019 Logical Access (LA)`** Zbiór referencyjny do detekcji syntetycznej i konwertowanej mowy. W projekcie wykorzystamy podzbiór nagrań prawdziwych do wygenerowania kontrolowanego zbioru z wstrzykniętymi artefaktami.
3. **`XSum` / `HC3` (dla wariantu tekstowego):** Publiczny korpus krótkich artykułów i wypowiedzi ludzkich.


## Ground Truth

1. **Ground Truth A (`PartialSpoof` – częściowo syntetyczna mowa + kontrola granic):**
   * **Definicja GT A:** Wykorzystujemy oryginalne nagrania ze zbioru `PartialSpoof` wraz z dostarczonymi znacznikami czasowymi segmentów syntetycznych. Ponieważ praca *Liu et al. (2024)* wykazała, że detektory mogą skupiać się na samych przejściach między segmentami, udział masy atrybucji będzie mierzony zarówno w całym oknie czasowym segmentu syntetycznego, jak i osobno w wąskich pasmach granicznych ($\pm 50\text{ ms}$ wokół punktów cięcia).
   * **Kontrola granic (`Real-into-Real Splices`):** Wygenerujemy zbiór kontrolny poprzez wklejenie fragmentów jednego prawdziwego nagrania w inne prawdziwe nagranie. Pozwoli to zmierzyć, w jakim stopniu detektor oraz mapy atrybucji reagują na sam artefakt montażu przy całkowitym braku mowy syntetycznej.

2. **Ground Truth B (`Planted Artifacts` – 4 sztuczne artefakty w 3 siłach + detektor kontrolny):**
   * **Definicja GT B:** Bierzemy wyłącznie prawdziwe nagrania ze zbioru `ASVspoof 2019 LA` i losowo dzielimy je na dwie równe klasy: Klasa 0 (nagrania czyste) oraz Klasa 1 (nagrania z wstrzykniętym w losowym, zapisanym oknie czasowo-częstotliwościowym artefaktem). Dzięki temu jedyną cechą różniącą obie klasy jest wstrzyknięty artefakt, co daje 100% gwarancji, że wytrenowany detektor osiągający wysoką dokładność musi z niego korzystać.
   * **4 typy artefaktów i 3 poziomy siły (Strengths):** Wstrzykniemy cztery zlokalizowane artefakty: (1) *wąskopasmowy ton sinusoidalny* (lokalny w czasie i częstotliwości), (2) *krótki impuls/trzask szerokopasmowy* (lokalny w czasie), (3) *lokalny szum wąskopasmowy* oraz (4) *lokalne tłumienie pasma / filtrację dolnoprzepustową*. Każdy artefakt zostanie wygenerowany na **3 poziomach intensywności (SNR względem sygnału mowy: słaby, średni, mocny)**, aby sprawdzić, czy artefakt nie jest zbyt trudny (ignorowany przez sieć) lub zbyt łatwy (trywialny dla każdej metryki).
   * **Kontrola (`Unplanted Control Detector`):** Wytrenujemy identyczną architekturę detektora na tym samym podziale nagrań *przed* wstrzyknięciem artefaktów, weryfikując, że dokładność oscyluje wokół losowego poziomu $50\%$, a mapy atrybucji nie wykazują pozornego skupienia w regionach docelowych.

3. **Ground Truth Text Variant:**
Do ludzkich tekstów wstrzykujemy kontrolowaną wskazówkę (np. specyficzne słowa funkcyjne/przysłówki), a jako transformację zachowującą semantykę stosujemy parafrazę modelem `T5`.


## Models

* **Wspólna siatka czasowo-częstotliwościowa:**
  Wszystkie analizowane nagrania audio ($16\text{ kHz}$) są przekształcane do wspólnej reprezentacji 2D **Log-Mel Spektrogramu**. Dzięki temu każda mapa atrybucji ma identyczną rozdzielczość czasowo-częstotliwościową, a operacje 2D (takie jak interpolacja sąsiadów w metodzie ROAD) są jednoznacznie zdefiniowane.
* **Detektor Audio:** `ResNet-18` trenowany na spektrogramach 2D
* **Modele dla Wariantu Tekstowego:**
  * Detektor - `roberta-base-openai-detector` (125M parametrów)
  * Model parafrazujący - `t5-base`


## Attribution Methods

Ocenimy 5 metod atrybucji XAI (z użyciem bibliotek `Captum`, `Zennit`, `Quantus`) wraz z losową jako kontrolną:

1. **Integrated Gradients (`IG`):** Ścieżkowa metoda gradientowa, która wyznacza ważność cech poprzez całkowanie gradientów wyjścia modelu wzdłuż linii prostej od neutralnego wejścia (ciszy/zer) do badanej próbki.
2. **SmoothGrad-Squared (`SG-SQ`):** Generuje $J$ kopii wejścia z dodanym szumem gaussowskim, wylicza dla nich mapy bazowe, a następnie podnosi je do kwadratu przed uśrednieniem.
3. **Grad-CAM:** Metoda oparta na aktywacjach wewnętrznych, która waży mapy cech ostatniej warstwy konwolucyjnej za pomocą uśrednionych gradientów klasy docelowej i nakłada funkcję ReLU.
4. **Layer-wise Relevance Propagation (`LRP`):** Metoda wstecznej propagacji istotności, która rozdziela wynik predykcji z warstwy wyjściowej wstecz przez kolejne warstwy sieci. Pozwala na precyzyjną analizę wkładu poszczególnych pikseli spektrogramu bez konieczności próbkowania perturbacji.
5. **Occlusion (Feature Ablation):** Metoda niezależna od gradientów. Maskuje przesuwające się okno czasowo-częstotliwościowe (lub kolejne tokeny w tekście) i mierzy bezpośredni spadek predykcji modelu.
6. **Random Map Control (`Random`):** Kontrolna mapa atrybucji generowana z rozkładu jednostajnego $\mathcal{U}(0, 1)$ całkowicie niezależnie od danych wejściowych i wag modelu. Służy jako dolny punkt odniesienia.


## Metrics

Na wyznaczonych mapach atrybucji uruchomimy następujące metryki:

1. **Ground Truth Recovery:** Mierzy udział całkowitej atrybucji (sumy wartości bezwzględnych mapy), który znajduje się wewnątrz znanego obszaru Ground Truth (wklejonego segmentu w GT A lub wstrzykniętego artefaktu w GT B). Definiuje bazowy ranking metod.
2. **Deletion Curves (MoRF - 3 wypełnienia: `Zero`, `Mean`, `Shaped Noise`):** Kolejno usuwa od najważniejszych cech ułamek pikseli spektrogramu, zastępując je zerami, średnią wartością lub dopasowanym szumem, i wyznacza pole pod krzywą spadku predykcji (AUC). Szybki spadek predykcji (niskie AUC) oznacza wysoką ocenę wierności mapy.
3. **Insertion Curves (LeRF / Keep - 3 wypełnienia: `Zero`, `Mean`, `Shaped Noise`):** Zaczynamy od całkowicie zamaskowanego spektrogramu (zerami, średnią lub szumem) i stopniowo odkrywamy najważniejsze cechy według mapy atrybucji. Szybki wzrost predykcji do wartości oryginalnej (wysokie AUC) świadczy o tym, że mapa poprawnie wskazała cechy wystarczające do klasyfikacji.
4. **Leak-free Imputation (`ROAD` - Noisy Linear Imputation):** Strategia wycinania cech, w której usunięte piksele spektrogramu są uzupełniane średnią ważoną sąsiadów z dodaniem małego szumu gaussowskiego. Zapobiega wyciekowi informacji o klasie przez kształt maski i utrzymuje próbkę w rozkładzie danych.
5. **Head-only Retraining (`ROAR-Head`):** Po zamaskowaniu top-t cech na zbiorze treningowym i testowym zamraża się ekstraktor cech i trenuje od nowa wyłącznie głowę klasyfikatora. 
6. **Infidelity (`INFD`):** Miara obliczająca oczekiwany błąd średniokwadratowy pomiędzy iloczynem skalarnym perturbacji i przeskalowanej mapy atrybucji a rzeczywistą zmianą wyjścia modelu na rozkładzie losowych zakłóceń. Niższa wartość oznacza lepsze liniowe przybliżenie zachowania modelu przez mapę.
7. **Max-Sensitivity (`SENSmax`):** Mierzy maksymalną normę zmiany znormalizowanej mapy atrybucji przy dodawaniu do wejścia drobnego szumu $\delta$ o promieniu $r$ niezmieniającego predykcji modelu. Ocenia odporność samego wyjaśnienia na lokalne zakłócenia.
8. **Reliability Tests on Metrics (`MetaQuantus` + MoRF/LeRF Consistency):** Testy meta-ewaluacyjne oceniające same metryki wierności bez udziału Ground Truth: badają spójność rozkładu ocen (`IAC`) oraz spójność rankingów metod (`IEC`), składające się na wynik **Meta-Consistency (`MC`)** w dwóch scenariuszach - odporności na mały szum wejścia/wag (*Noise Resilience - NR*) oraz reaktywności na celowe psucie modelu (*Adversary Reactivity - AR*), uzupełnione o test zgodności kierunków MoRF vs. LeRF.


### Sposób porównania i warunek obalenia hipotezy
Dla każdego zbioru (GT A, GT B na 3 poziomach siły oraz Text Variant) wyznaczymy współczynniki korelacji (Spearmana, Kendalla) między rankingiem 5 metod XAI wyznaczonym przez każdą z badanych metryk a rankingiem referencyjnym z *Ground Truth Recovery* (oraz macierz korelacji metryk między sobą), raportując 95% przedziały ufności wyznaczone metodą *bootstrap* po klipach audio.

**Obalenie hipotezy:** Hipoteza o niezgodności metryk i wpływie artefaktów wycinania w detektorach audio zostanie obalona, jeśli wszystkie badane metryki (w tym zwykłe *zero-fill deletion*, *ROAD*, *Head-only Retraining* oraz *Infidelity*) uszeregują 5 metod atrybucji identycznie oraz osiągną zgodność z Ground Truth mieszczącą się we wspólnych przedziałach ufności bootstrap dla obu zbiorów (GT A i GT B) na każdym z trzech poziomów siły artefaktu.


