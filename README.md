# FORGE AI Android — integração de Transformer nativo (prévia técnica)

Projeto Android Kotlin. O aplicativo inclui **inferência nativa em CPU do Transformer causal** e pesos treinados do próprio projeto (`forge-transformer.frg`). Não utiliza APIs de IA nem modelos de terceiros. O checkpoint incluído foi treinado somente como teste e **não é adequado para conversação inteligente ou geração confiável de código**.

## Compilar APK

No GitHub, publique os arquivos na raiz de um repositório e execute Actions → Android APK → Run workflow. O artifact gerado será `forge-ai-debug-apk`. Para compilar localmente, instale JDK 17, Android SDK 35 e Gradle 8.7, depois execute `gradle assembleDebug`.

## Treinar e atualizar o modelo Android

```bash
pip install -r training/requirements.txt
python training/forge_transformer.py train --data training/seed_corpus.txt --width 32 --layers 1 --heads 4 --context 32 --steps 100 --output training/checkpoints/custom.pt
python training/export_android.py --checkpoint training/checkpoints/custom.pt --output app/src/main/assets/forge-transformer.frg
```

A implementação Kotlin suporta o Transformer byte-level com LayerNorm, atenção causal, feed-forward GELU e amostragem por temperatura. Não há geração garantida de scripts. A biblioteca de código fornece templates estáticos separados da geração neural.

## Limitações

- O treinamento em escala grande **não foi realizado**.
- O APK **não foi compilado ou testado em aparelho neste ambiente**.
- A saída do modelo pequeno é experimental e pode não ser coerente.
- Chaves de proprietário são locais e não fornecem autorização de servidor.


## Estado do repositório (2026-11-20)

A estrutura de código Android (`app/`), treinamento (`training/`), configuração (`config/`) e workflow `.github/workflows/android.yml` foi reorganizada. **Os arquivos binários de pesos ainda não foram publicados**: `app/src/main/assets/forge-transformer.frg` e `app/src/main/assets/forge-seed-weights.bin`. O aplicativo deve informar que o modelo não está treinado quando não houver pesos, e os templates locais continuam disponíveis. O workflow de APK foi adicionado, mas seu resultado ainda precisa ser verificado. Para habilitar inferência Transformer, adicionar os pesos exportados do projeto em `app/src/main/assets/`.
