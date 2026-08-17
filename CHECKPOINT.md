# Dynamo Snapshot — checkpoint di lavoro

Data: 2026-08-17 UTC

## Obiettivo

Ridurre il tempo osservato dall'input al primo token per un worker Dynamo
Snapshot ripristinato, mantenendo PVC autorevole, checkpoint immutabili,
canary reale prima del routing e nessuna modifica a vLLM/modello/driver.

## Misure confermate

| Percorso | CRIU | Totale agent |
| --- | ---: | ---: |
| PVC freddo | 38,70 s | 44,14 s |
| Cache node-local | 17,07 s | 21,53 s |
| tmpfs, percorso stabile | ~10–11 s | ~14,5–15,3 s |

- Wake KV/vLLM: ~0,39 s.
- Primo request su engine gia' sveglio: ~1,64 s.
- Payload checkpoint: 21.834.939.790 byte (20,335 GiB), 384 file.
- Immagini CRIU `pages-*`: 20,292 GiB; `pages-13.img`: 18,093 GiB.
- Rootfs diff: 31,26 MiB; metadata: ~14 MiB.
- Lettura buffered isolata di 18,09 GiB su tmpfs: 1 reader 11,67 s;
  8 reader 1,79 s.
- PVC: AIO Q=128 e' gia' saturo; Q=512 non produce miglioramento utile.

## Diagnosi attuale

Il limite dominante su PVC e' il percorso storage e la ricostruzione CRIU
delle pagine, non la GPU o la CPU sature. Durante il restore freddo sono stati
misurati ~440 MiB/s, ~11,31% CPU aggregata e ~1,6% GPU media.

L'integrazione dei reader buffered paralleli ha rivelato una race reale:
restorer di processi distinti eseguono in parallelo; un reader temporaneo di
un processo puo' occupare un PID/TID che un altro restorer deve ricreare con
`clone3(set_tid)`. L'evidenza e' `EEXIST` con il TID richiesto ancora vivo.

## Correzione implementata e validata

- I reader sono helper process reaped esplicitamente, non thread CRIU.
- Una barriera futex condivisa deve far attendere ogni task leader dopo il
  page restore/reap e prima della ricreazione dei thread applicativi.
- La barriera conta solo `task_alive()`: zombie e helper non partecipano.
- Un abort deve risvegliare la barriera da ogni percorso d'errore.
- `wait4(...)=ECHILD` per un reader deve essere un errore fail-stop, non un
  successo presunto.

## Test gia' aggiunto

`poc/regolo-vllm-snapshot/experimental/ghost-kv/tests/test_multiprocess_restore_barrier.py`

- 3 processi, 16 pthread ciascuno;
- 320 MiB PRNG incomprimibili per processo;
- dump/restore CRIU buffered;
- checksum, heartbeat e PID/TID verificati prima/dopo;
- richiede evidenza di almeno 8 reader per almeno due processi.

Il test e' opt-in e privilegiato: `CRIU_MULTIPROCESS_REGRESSION=1` e
`CRIU_BIN` devono puntare al build CRIU candidato.

## Risultato finale del ciclo

- La patch CRIU con barriera fail-stop e' stata costruita in un builder Jammy
  compatibile con il target ed ha superato 3/3 regressioni privilegiate
  multiprocesso.
- Il restore buffered a 8 reader da tmpfs ha ridotto CRIU a 3,42--4,87 s.
- Il restore CUDA, prima seriale per quattro PID, ora puo' usare in modo opt-in
  una scala limitata di 2 o 4 worker, senza unlock finche' tutti i restore non
  hanno avuto successo.
- Con quattro worker: CRIU 3,424 s, CUDA 4,883 s, restore agent 8,514 s,
  wake 0,396 s e primo byte del canary 0,091 s: **9,001 s** come somma delle
  fasi dal rilevamento del restore al primo byte, con canary semantico valido e
  nessun Xid/OOM.

Questo numero non include scheduling, creazione del placeholder/pod o la sua
inizializzazione: da un pod totalmente assente al primo token il budget misurato
resta circa 11--13 s. I pesi/checkpoint non sono stati modificati; la strada
sotto 10 s richiede tenere disponibile il checkpoint locale (tmpfs/RAM) e un
placeholder gia' pronto al restore.

## Vincoli invariati

- PVC e checkpoint esistenti non vengono modificati o cancellati.
- Nessun worker entra nel routing prima del canary valido.
- Fallback al PVC se una cache locale non e' valida.
- Nessun GMS, weight reservoir, GDS o modifica del formato/modello in questo
  ciclo di ottimizzazione.
