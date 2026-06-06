# Odtworzenie wyników pracy KnockOffNets

https://arxiv.org/pdf/1812.02766

## Struktura kodu

Aby uruchomić trening sieci bazowych uruchom plik

```
uv run model_steal/train_baseline.py
```

Aby uruchomić generowanie logitów sieci bazowej uruchom plik

```
uv run model_steal/infer_logits_for_baseline.py
```

Aby uruchomić trening sieci knock-off uruchom plik

```
uv run model_steal/train_knock_off.py
```

Aby ewaluować jakość wszystkich modeli uruchom plik
```
uv run model_steal/knock_off_eval.py
```

Dodatkowe grafiki z trenowania można wygenerować uruchamiając skrypt

```
uv run model_steal/tran_stats_plot.py
```