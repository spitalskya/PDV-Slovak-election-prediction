# Navrh modelu

- data
  - riadky v tvare: [strana]-[volebny rok], vysledok v danych volbach, prieskum 1 mesiac pred volbami, ...., prieskum n mesiacov pred volbami
  - medizmesacne rozdiely (rozdiely/nasobky?): [strana]-[volebny rok], vysledok v danych volbach, narast/pokles medzi prvym a druhym mesiacom pred volbami, ...., narast/pokles medzi (n-1)-vym a n-tym mesiacom pred volbami
  - charakteristiky o stranach pred nejakymi volbami: pravica/lavica, konzervativ/liberal, vo vlade, v opozicii, (mimo parlamentu), pokles/narast HDP za volebne obdobie (nasobok?), ...
    - v modeli robit interakcie premennych `vo vlade` * `pokles/narast zivotnej urovne` 
- model
  - dogenerovat data - simulovat viacere prieskumne agentury, pripocitat k datam `runif` z intervalu spolahlivosti
  - rozdelit data do skupin na zaklade prieskumov - strany s podobnymi trendami pred volbami
    - unsupervised clustrovanie
    - regresne stromy
    - ...
  - linearna regresia na datach charakteristickych pre strany, potencialne aj posledny prieskum pred volbami
    - bud `y` je volebny vysledok alebo rozdiel volebneho vysledku s poslednym prieskumom
    - chceme predikovat ako sa stranam dari vo volbach voci poslednemu prieskumu
- predikcie:
  - dojde SMER-2024 s prieskumami za posledny rok od volieb
  - treba ho zaradit do clustru
  - pouzit `beta` pre ten cluster
-  baseline predikcie
  - musi byt lepsi ako predikcia vvolieb = prieskum

- rozsirenie 
  - rozne modely pre roznu volebnu ucast (mame dostatocne data na to?)
