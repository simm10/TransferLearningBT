# TransferLearningBT
Repozitár obsahuje finálne zdrojové kódy, ktoré boli použité pri realizácií bakalárskej práce.

Skript Run.py obsahuje metódy na trénovanie a testovanie navrhnutých architektúr
Pred spustením je nutné upraviť skript na miestach označených ako "Change it"
  * DATASET_DIR - cesta k datasetu, tu sú dve možnosti podľa voľby binárnej alebo viactriednej klasifikácie, je potrebné zadať správnu
                cestu ku správnemu datasetu
  * THIS_DIR - plná cesta, kde je uložený súbor run.py

Zároveň je nutné mať vedľa súboru run.py zložku nazvanú logs a zložku nazvanú models, kam sa umiestnia pretrénované modely, prípadne kam sa bude ukladať model počas procesu trénovania. Tie je možné stiahnúť zo stránok portálu [ulož.to](https://uloz.to/tam/_zTq0gGFe1sdz). 
Celý obsah je potrebné stiahnúť a všetky súbory, okrem zložky logs, vložiť do zložky models. 
Štruktúra je následovná: 
 * binaryWeights.h5 - váhy izolovaného binárneho modelu
 * binaryWeightsTl.h5 - váhy binárneho modelu s preneseným učením
 * BB.h5 - váhy izolovaného viactriedneho modelu
 * MTLB.h5 - váhy viactriedneho modelu s preneseným učením
 * logs - štrukturovaný zip súbor so záznamami TensorBoard
  * logsBinary - záznamy binárnych modelov v TensorBoard
  * logsMulticlass - záznamy viactriednych modelov v TensorBoard
Zip súbor logs obsahuje záznami z trénovania jednotlivých modelov ako boli prezentované v texte práce.
  
Skript sa spúšťa z príkazovej riadky príkazom: 
  * python run.py \<model\> <operácia>
  
dostupné možnosti pre hodnotu \<model\>:
  * binary - pre modely binárnej klasifikácie
  * multiclass - pre modely viactriednej klasifikácie
  
dostupné možnosti pre hodnotu <operácia>:
  * train - spustí proces trénovania na modely bez preneseného učenia
  * test - spustí proces testovania na modely bez preneseného učenia
  * train_tl - spustí proces trénovania na modely s preneseným učením
  * test_tl - spustí proces testovania na modely s preneseným učením
