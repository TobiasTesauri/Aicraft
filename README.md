# Aicraft – Framework di Machine Learning a Basso Impatto

<div align="center">
  <img src="./logo.png" alt="T&M Softwares Logo" width="300"/>
</div>

**Aicraft** è un framework di machine learning scritto interamente in C, pensato per operare su sistemi a risorse limitate (meno di **512 MB** di memoria) e capace di sfruttare il backend CUDA in modo opzionale.  
Creato da **Tobias Tesauri**, Aicraft nasce come sfida personale per comprendere le fondamenta del deep learning senza ricorrere a librerie esterne.

---

## Indice

- [Introduzione](#introduzione)
- [Caratteristiche](#caratteristiche)
- [Architettura](#architettura)
- [Getting Started](#getting-started)
- [Compilazione e Utilizzo](#compilazione-e-utilizzo)
- [Estensioni Future](#estensioni-future)
- [Documentazione](#documentazione)
- [Licenza](#licenza)
- [Contatti](#contatti)


---

## Caratteristiche

- **Zero dipendenze:** Intero codice scritto in C, senza reliance su librerie esterne.
- **Supporto duale:** Compatibilità con CPU e, opzionalmente, accelerazione via CUDA.
- **Memory-friendly:** Progettato per operare con un utilizzo della memoria inferiore a 512 MB.
- **Design Modulare:** Separazione chiara tra gestione tensori, training e gestione dei backend.
- **Logging & Profiling:** Output colorato e configurabile, per monitorare l’esecuzione in tempo reale.
- **Didattico & Innovativo:** Ogni funzione matematica è stata derivata “on the fly”, enfatizzando la comprensione profonda.

---

## Architettura

Aicraft è strutturato in vari moduli principali:

- **Tensor Engine:** Gestisce l’allocazione, l’elaborazione e le operazioni aritmetiche sui tensori.
- **Modulo Training:** Implementa la forward propagation e contiene gli ottimizzatori (es. SGD, Adam).
- **Gestore Backend:** Rileva la presenza di CUDA e, se disponibile, la usa; altrimenti, ricade sulla CPU.
- **Logger & Profiler:** Fornisce un sistema di log dettagliato e strumenti di profilazione per il monitoraggio 
  delle performance.

**Diagramma Architetturale:**

+-----------------------+ | Input Data | +-----------+-----------+ | v +-----------------------+ | Tensor Engine | | (Allocazione & Ops) | +-----------+-----------+ | v +-----------------------+ | Modulo Training | | (Forward & Optim.) | +-----------+-----------+ | v +-----------------------+ | Gestore Backend | | (CPU / CUDA) | +-----------------------+


---

## Getting Started

### Prerequisiti

- **Compilatore C:** GCC (o qualsiasi compilatore compatibile con C99)
- **CUDA (opzionale):** Necessario se si intende sfruttare l'accelerazione GPU

### Installazione

Clona il repository:

```bash
git clone https://github.com/TobiasTesauri/Aicraft.git
cd Aicraft
Compilazione e Utilizzo
Compilazione
Puoi compilare il progetto manualmente:

bash
gcc -std=c99 -o aicraft src/*.c -lm
Oppure, se presente, utilizza il Makefile:

bash
make all
Esecuzione
Avvia il programma:

bash
./aicraft
Note: Assicurati di avere la configurazione adatta se utilizzi accelerazione CUDA.

Estensioni Future
Backpropagation Completo: Integrazione di un sistema completo per il retropropagazione.

Quantizzazione: Supporto a computazioni in int8/fixed-point per dispositivi embedded.

Serializzazione dei Modelli: Funzionalità per salvare e caricare modelli.

Interfaccia CLI/API: Sviluppo di un’interfaccia user-friendly per semplificare l’interazione col framework.

Contributi e suggerimenti sono sempre benvenuti!

Documentazione
La documentazione completa, inclusi whitepapers con le derivazioni matematiche e guide tecniche, è disponibile nella cartella /docs.

Licenza
Questo progetto è rilasciato sotto licenza MIT.

Contatti
Tobias Tesauri

GitHub: TobiasTesauri

Email: tobias.tesaur@cillarioferrero.it

Telefono: 351 550 7405

Crafted with passion by T&M Softwares. Eleva la tua ricerca in AI con semplicità e innovazione.
