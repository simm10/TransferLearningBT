# TransferLearningBT
Repozitár obsahuje finálne zdrojové kódy, ktoré boli použité pri realizácií bakalárskej práce ako aj záznamy TensorBoard.

Skript Run.py obsahuje metódy na trénovanie a testovanie navrhnutých architektúr
Pred spustením je nutné upraviť skript na miestach označených ako "Change it"
  * DATASET_DIR - cesta k datasetu, tu sú dve možnosti podľa voľby binárnej alebo viactriednej klasifikácie, je potrebné zadať správnu
                cestu ku správnemu datasetu
  * THIS_DIR - plná cesta, kde je uložený súbor run.py

Zároveň je nutné mať vedľa súboru run.py mať zložku nazvanú models, kam sa umiestnia pretrénované modely, prípadne kam sa bude ukladať model počas procesu trénovania. Tie je možné stiahnúť z nižšie uvedených odkazov
  * [izolovaný model binárnej klasifikácie](https://www.google.com)
  * [model binárnej klasifikácie s preneseným učením](https://www.google.com)
  * [izolovaný model viactriednej klasifikácie](https://www.google.com)
  * [model viactriednej klasifikácie s preneseným učením](https://www.google.com)
  
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
