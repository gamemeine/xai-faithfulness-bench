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

| Tydzień             | Deadline | Opis |
| ------------------- | -------- | ---- |
| #1 (22.10 - 28.10)  |  28.10 - design proposal        | design proposal, w tym wstępny przegląd załączonej literatury oraz plan projektu     |
| #2 (29.10 - 04.11)  |          | pierwsze kroki, w tym uruchomienie bazowego detektora (todo: którego???) na 100 próbkach PartialSpoof oraz 4 metodach (todo: jakich???), zebranie map wyjaśnień. Dalszy przegląd literatury.   |
| #3 (05.11 - 11.11)  |  6.11 - prototyp deadline na spotkanie        |  **Compute deletion curves with zero fill and with noise fill and compare the method rankings.** Prezentacja wyników jako prototyp. Porównanie wstępnych rankingów metod.    |
| #4 (12.11 - 18.11)  |          | Przygotowanie ground truth A na podstawie segmentów PartialSpoof, ground truth B generowanie sztucznych artefaktów + próbki kontrolne.     |
| #5 (19.11 - 25.11)  |          | Nauka na kolokwium.     |
| #6 (26.11 - 02.12)  |          | Trenowanie detektorów na danych z artefaktami oraz modelu kontrolnego, sprawdzenie, czy modele rzeczywiście wykorzystują planted features.  |
| #7 (03.12 - 09.12)  |          |      |
| #8 (10.12 - 16.12)  |          |      |
| #9 (17.12 - 23.12)  |          |      |
| #10 (24.12 - 30.12) |          |      |
| #11 (31.12 - 06.01) |          |      |
| #12 (07.01 - 13.01) |          | spisanie finalnej wersji raportu, poprawki      |
| #13 (14.01) | 14.01 - oddanie projektu         |      |


## Related work

.... todo


