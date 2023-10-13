# Muudetud M2 kontrollija

Tegu on eesti õppijakeele märgenduse jaoks modifitseeritud versiooniga [M2 Scorerist](https://github.com/nusnlp/m2scorer).  
Sisse on viidud kaks muudatust:
1. Arvestamine eesti õppijakeele [M2 märgendusel](https://github.com/tlu-dt-nlp/m2-corpus) kasutatud sõnajärjevea (*word order* e. *WO*) märgendiga, mis erinvealt algsest M2 märgendusskeemist võib oma skoobis sisaldada teisi vigu.
2. Lisastatistika väljatrükk täpsusele ja saagisele sõnaliigiti.

## Word Order e. sõnajärjeviga
Kuna word order on edit, mis hõlmab ka teisi, aga need seal sees võivad olla eraldi märgendatud-parandatud, tuleb sellega eraldi tegelda. WO veaparanduses on juba kõik muud vead ka ära parandatud, nt.

```
S Mul ei olnud kassi mitte kunagi .  
A 2 3|||R:VERB:FORM|||ole olnud|||REQUIRED|||-NONE-|||0  
A 2 6|||R:WO|||ole mitte kunagi kassi olnud|||REQUIRED|||-NONE-|||0
```

Kus on 2-3 verbivorm teine "olnud"->"ole olnud", aga samas on täiesti korrektses lauses "ole" ja "olnud" parema sõnajärje huvides lahku löödud: 2-6 "olnud kassi mitte kunagi"->"ole mitte kunagi kassi olnud".

### Täpsus ja saagis
Praegu on see lahendatud nii, et tehakse (m2scorer.py failis) juurde pseudomärgendaja. Seega jääb alles:  
\* algne märgendaja 0, kellel on 2 parandust (2-3 verbivorm ja 2-6 sõnajärg),  
lisaks tekib  
\* märgendaja 1, kellel on vahemiku 2-6 jaoks vaid üks suur sõnajärjeparandus (2-6 sõnajärg)  
\* märgendaja 2, kellel on vahemiku 2-6 jaoks alles vaid pisemad parandused (2-3 verbivorm - siia läheksid ka muud, kui neid siin oleks)

See lähenemine ei arvesta asjaoluga, et märgendaja 2 puhul jääb sõnajärg parandamata - selle võrra on meil täpsus pisut kõrgem. Samas on üsna vähetõenäoline, et tehakse parandus, kus on ära muudetud kõik pisivead, aga pole muudetud sõnajärge.  
Samuti arvestavad täpsus-saagis märgendaja 1 puhul üksnes ühe parandatud veaga, kuigi tegelikult oli neid seal märksa rohkem. Tehniliselt võiks sel juhul arvestada korrektselt parandatute hulka ka kõik pisivead, aga sel juhul läheksime pisut teisele poole kiiva. Üldine M2 skoor on niikuinii nihkes, kuna täpsust loetakse paranduste, mitte lausete pealt - ja seda vastavalt parima sobivusega märgendaja parandustele. Kuna eri märgendajad võivad teha erineva arvu parandusi, siis on nn kogusumma kõikumine siia skoori sisse arvestatud. Samast põhimõttest lähtudes võiks märgendaja 1 praegune kuju olla parem kui välja toodud alternatiiv.  
Märgendaja 0 ei saa meie märgendusskeemi järgi kunagi 100% täpsusega parandada, kuna seal sees olevad parandused lähevad põhimõttteliselt omavahel konflikti.

## Vealiigiti
Vealiigiti arvestamisel on vaadatud samuti pakutud paranduste kokkulangevust sobivaima märgendajaga. Kuna masin hetkel vealiiki ei paku, siis vealiigistatistika töötab üksnes saagise pealt: kui palju (sobivaimate märgendajate) vastava vealiigiga parandustest üles leiti ja korrektselt ära parandati.  
Selleks et anda infot ka juhtude kohta, mil tuvastati küll õiges kohas viga, aga selle parandamisel eksiti, on kolme eri liiki statistikat. Iga vealiigi kohta väljastatakse (tabulaatoriga eraldatult):  
\* täiesti õigete paranduste %  
\* korrektse veatuvastuse % ehk kui sageli tuvastati viga täpselt samas vahemikus  
\* osalise veatuvastuse % ehk kui sageli kattus vähemalt osaliselt selle parandusega vähemalt üks tuvastatud viga.

Kuna sõnajärjevea täielik tuvastamine ja korrektne parandus hõlmab oma praegusel kujul teiste vealiikide parandusi (vt 'ole olnud' eelmises näites), siis on mõttekas seda eraldi statistikas arvestada. Seega näitelause *"Mul ei olnud kassi mitte kunagi ."* puhul võiks pakutud parandus *"Mul ei ole mitte kunagi kassi olnud ."* anda tulemuseks ka teadmise, et kaetud on nii sõnajärjeviga (R:WO) kui ka seal sees peitunud verbivormiviga (R:VERB:FORM). Verbivormivigade saagise statistikasse võiks see vastus korrektse parandusena sisse minna. Seega on lisaks eelmistele näitajatele omakorda kolm statistikut, mis arvestavad ka sõnajärjevigade skoobi sees olnud vealiikidega ja loevad need korrektselt parandatuks, kui terve WO oli korrektselt parandatud. Korrektse WO skoobi puhul loetakse seesolnud vealiik osaliselt tuvastatuks, muul juhul kontrollitakse eraldi üle.