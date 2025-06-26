<div align="center">
  <img src="./logo.png" alt="Aicraft Logo" width="300"/>
  <h1>Aicraft</h1>
  <h3>Machine Learning Framework in C – Per chi vuole davvero capire (e soffrire)</h3>
  <p>
    <a href="https://github.com/TobiasTesauri/Aicraft"><img src="https://img.shields.io/github/stars/TobiasTesauri/Aicraft?style=social" alt="Stars"></a>
    <a href="https://github.com/TobiasTesauri/Aicraft/blob/main/LICENSE"><img src="https://img.shields.io/github/license/TobiasTesauri/Aicraft?color=blue" alt="License"></a>
    <img src="https://img.shields.io/badge/C-99-blue.svg" alt="C99">
    <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20Embedded-lightgrey">
    <img src="https://img.shields.io/badge/Backend-CPU%20%7C%20CUDA-green">
  </p>
  <blockquote>
    <b>Perché usare TensorFlow è troppo facile... e troppo noioso.</b>
  </blockquote>
</div>

---

## 🚀 Introduzione

Se stai leggendo qui, probabilmente sei stufo delle solite “magic box” che fanno deep learning col bottone “train”.  
**Aicraft** è un framework scritto in C puro, minimale e spietato. Nessuna dipendenza, nessun layer astratto dietro cui nascondere l’incompetenza: solo C, puntatori, e una sana dose di profanità.  
Pensato per chi vuole capire davvero cosa succede sotto il cofano – o per chi ama complicarsi la vita.

- **Autore:** Tobias Tesauri (che evidentemente non aveva nulla di meglio da fare)
- **Obiettivo:** Rendere il machine learning trasparente… e un po’ più doloroso
- **Per chi:** Studenti masochisti, ingegneri old school, nostalgici di malloc, e chi pensa che Python sia “troppo user-friendly”.

---

## 📑 Indice

- [Perché dovresti (non) usarlo](#perché-dovresti-non-usarlo)
- [Caratteristiche (desolate)](#caratteristiche-desolate)
- [Architettura (inutilmente chiara)](#architettura-inutilmente-chiara)
- [Componenti principali](#componenti-principali)
- [Esempi (che funzionano, si spera)](#esempi-che-funzionano-si-spera)
- [Installazione & Quickstart (o Quickragequit)](#installazione--quickstart-o-quickragequit)
- [Compilazione (divertiti con il Makefile)](#compilazione-divertiti-con-il-makefile)
- [Come Funziona (spoiler: a fatica)](#come-funziona-spoiler-a-fatica)
- [Roadmap (sogni irrealizzabili)](#roadmap-sogni-irrealizzabili)
- [Contribuire (se proprio insisti)](#contribuire-se-proprio-insisti)
- [FAQ (Frequently Annoying Questions)](#faq-frequently-annoying-questions)
- [Documentazione (per chi legge davvero)](#documentazione-per-chi-legge-davvero)
- [Licenza](#licenza)
- [Contatti & Credits](#contatti--credits)

---

## 🦴 Perché dovresti (non) usarlo

- Vuoi capire cosa succede davvero in una rete neurale, compresi tutti i bug.
- Ti piace il brivido di gestire la memoria a mano.
- Odii le dipendenze.
- Vuoi performance su hardware miserabile.
- Ti sei stancato di “pip install” e vuoi sudare per ogni riga compilata.

Se invece vuoi solo far girare il prossimo LLM, chiudi questa pagina. Seriamente.

---

## 🌟 Caratteristiche (desolate)

- **Zero dipendenze:** solo C99, niente scorciatoie.
- **CPU + CUDA:** se hai una GPU, bene; se no, arrangiati.
- **Memory friendly:** gira anche su tostapane, a patto di non chiedergli troppo.
- **Design modulare:** o almeno ci prova.
- **Logging colorato:** così puoi vedere gli errori in technicolor.
- **Codice didattico:** se lo capisci, sei già avanti.
- **Estensibilità:** puoi modificarlo, se ne hai il coraggio.
- **Whitepaper e guide:** per chi pensa che i commenti siano meglio dei tutorial su TikTok.

---

## 🏗️ Architettura (inutilmente chiara)

```mermaid
graph TD;
    InputData[Input Data] --> TensorEngine[Tensor Engine<br/>(Allocazione & Ops)];
    TensorEngine --> Training[Training Module<br/>(Forward, Backward, Ottimizzatori)];
    Training --> Backend[Backend Manager<br/>(CPU/CUDA)];
    Backend --> Logger[Logger & Profiler];
    Logger --> Output[Output & Profiling];
```
- **Tensor Engine:** i tuoi dati, la tua memoria, i tuoi crash.
- **Training Module:** qui si fa sul serio (o si segfaulta).
- **Backend Manager:** CPU o GPU? Dipende dal karma.
- **Logger & Profiler:** per capire quanto lentamente sta andando.

---

## 🧩 Componenti principali

### 1. Tensori
- Array multidimensionali con la personalità di un C struct.
- Funzioni per allocazione, slicing e per perdere la testa.

### 2. Modello
- API per reti sequenziali. Layer? Quelli che ti servono, o che riesci a scrivere.

### 3. Ottimizzatori & Loss
- SGD e Adam (in arrivo, forse). Loss personalizzabili, o semplicemente frustranti.

### 4. Backend
- CPU/GPU, a seconda dell’umore della tua macchina.

### 5. Logging & Debug
- Output colorato per i tuoi errori. Profiler per misurare quanto lentamente va tutto.

---

## 🧪 Esempi (che funzionano, si spera)

```c
#include "aicraft.h"

// Creazione tensore 3x3
Tensor* t = tensor_create(3, 3);
tensor_fill(t, 0.0f);

// Definizione modello semplice
Model* m = model_create();
model_add_dense(m, 3, 4, RELU);
model_add_dense(m, 4, 1, SIGMOID);
model_compile(m, MSE, SGD);

// Training su dati ridicoli
model_train(m, X_train, y_train, epochs=100, batch=10, learning_rate=0.01);

// Inference
Tensor* pred = model_predict(m, X_test);

// Pulizia memoria (obbligatoria, qui non c'è garbage collector)
model_free(m);
tensor_free(t);
```
*Altri esempi (e bug) nella cartella `/examples`.*

---

## ⚡ Installazione & Quickstart (o Quickragequit)

### Prerequisiti

- GCC (C99) o equivalente
- (Opzionale) CUDA Toolkit, se vuoi illuderti di avere performance

### Clonazione

```bash
git clone https://github.com/TobiasTesauri/Aicraft.git
cd Aicraft
```

---

## 🛠️ Compilazione (divertiti con il Makefile)

### Manuale

```bash
gcc -std=c99 -o aicraft src/*.c -lm
```

### Con Makefile

```bash
make all
```

### Esecuzione

```bash
./aicraft
```
> ⚠️ Se usi CUDA, buona fortuna.

---

## 🔎 Come Funziona (spoiler: a fatica)

1. **Definisci il modello**  
   Layer per layer, a mano. Nessun wizard.

2. **Prepara i dati**  
   Tensori, batch e un po’ di pazienza.

3. **Compila e addestra**  
   Scegli ottimizzatore e parametri. Speriamo bene.

4. **Valuta**  
   Se i risultati fanno schifo… benvenuto nel machine learning vero.

5. **Profiling**  
   Guarda i log e chiediti perché ci hai messo così tanto.

6. **Personalizza**  
   Perché sicuramente vorrai aggiungere “quel layer in più”.

---

## 🧭 Roadmap (sogni irrealizzabili)

- [ ] Backpropagation che non si schianta
- [ ] Quantizzazione (se non perdi prima la pazienza)
- [ ] Serializzazione modelli (per esportare le tue delusioni)
- [ ] API/CLI user-friendly (ma non troppo)
- [ ] Layer convoluzionali (CNN per chi crede ancora nelle immagini)
- [ ] Esempi per microcontrollori (se vuoi soffrire anche su embedded)
- [ ] Test automatici (magari uno che passa, prima o poi)
- [ ] Benchmarking (per vedere quanto sei lento rispetto a PyTorch)
- [ ] Video tutorial (ma chi li guarda davvero?)

---

## 🤝 Contribuire (se proprio insisti)

Pull request, issue e suggerimenti sono benvenuti. Se hai voglia di perdere tempo, accomodati.  
**Regole non scritte:**
- Scrivi codice chiaro (più o meno).
- Commenta, o almeno prova a spiegare l’inspiegabile.
- Testa su almeno una piattaforma (Linux, Windows, o sul tuo tostapane).

> 💬 Per dubbi esistenziali, apri una [Discussion](https://github.com/TobiasTesauri/Aicraft/discussions).

---

## ❓ FAQ (Frequently Annoying Questions)

**Q: È adatto per produzione industriale?**  
A: Solo se odi la tua azienda.

**Q: Supporta reti neurali complesse?**  
A: Più o meno, finché non esaurisci la RAM (o la pazienza).

**Q: Come posso aggiungere nuovi layer/ottimizzatori?**  
A: Scrivi codice, bestemmia, ripeti.

**Q: Posso usarlo su microcontrollori?**  
A: Sì, se ti piace l’avventura.

**Q: Si può esportare un modello?**  
A: In futuro. Per ora, carta e penna.

**Q: Tutorial video?**  
A: Forse, ma non aspettarti uno youtuber.

---

## 📚 Documentazione (per chi legge davvero)

La documentazione dettagliata è in `/docs`, per i pochi che ancora la consultano.  
Troverai:
- Derivazioni matematiche (per veri nerd)
- Esempi (testati, non garantiti)
- Tutorial scritti, niente TikTok

---

## 📝 Licenza

MIT. Fanne quello che vuoi. Non lamentarti se poi non funziona.

---

## 📞 Contatti & Credits

- **Autore:** Tobias Tesauri
- **GitHub:** [@TobiasTesauri](https://github.com/TobiasTesauri)
- **Email:** tobias.tesaur@cillarioferrero.it
- **Telefono:** 351 550 7405

> Prodotto con sudore, caffeina e un pizzico di disprezzo per le scorciatoie.

---

<div align="center">
  <b>⭐ Dai una stella se ti è piaciuto. O anche solo per compassione. ⭐</b>
</div>
