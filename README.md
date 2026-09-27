> ESTE É UM MODELO EXPERIMENTAL DE PROJEÇÃO ESTATÍSTICA. NÃO É UMA PESQUISA ELEITORAL, NEM REGISTRADA NO TSE [Res. 23.600](https://www.tse.jus.br/legislacao/compilada/res/2019/resolucao-no-23-600-de-12-de-dezembro-de-2019).
Não deve ser divulgado como pesquisa.

# Votum

Votum is an open-source experimental neural network for modeling Brazilian elections.

It combines spatial, temporal, socioeconomic, historical election, and polling data to estimate electoral distributions.

The repository contains the training and inference code used by Votum. The architecture and configurations are intentionally editable: you can replace the datasets, change which features are used, modify the candidates, scale the model, or adapt the pipeline to a different electoral scenario.

«Votum produces statistical/model estimates from the data provided to it. Its outputs are not official polls and should not be interpreted as guaranteed election results.»

## Architecture

Votum combines three main components:

``Municipal / historical data
          │
          ▼
 GraphSAGE + GATv2
     Spatial GNN
          │
          ├──────────────┐
                         ▼
Polling / temporal → Transformer
      data               │
                         ▼
                  Cross-Attention
                         │
                         ▼
                   MC Dropout
                         │
                         ▼
              Electoral Distribution``

## Spatial GNN

Municipal information is represented as a geographic graph.

The default model starts with a GraphSAGE layer followed by GATv2 layers:

``Municipal features
       ↓
   GraphSAGE
       ↓
     GATv2
       ↓
     GATv2
       ↓
Spatial representation``

The graph can incorporate features such as:

- population;
- electoral density;
- income;
- literacy;
- HDI;
- historical election results;
- geographic relationships between municipalities.

The default implementation uses a geographic k-nearest-neighbor graph, but this can be replaced with another graph construction strategy.

## Temporal Transformer

Electoral signals evolve over time.

Votum therefore uses a Transformer to process temporal series for each state.

These series can contain, for example:

``Polling data
Economic indicators
Interest rates
Inflation
Other temporal variables
        ↓
   Transformer
        ↓
Temporal representation``

The exact features are configurable.

## Cross-Attention

The spatial and temporal representations are combined through cross-attention.

``Spatial representation ──┐
                         ├── Cross-Attention → Prediction
Temporal representation ─┘``

This allows the model to combine geographic electoral structure with changing signals such as polling data.

## Probabilistic Head

The final prediction head uses dropout and can perform multiple Monte Carlo samples during inference.

This allows Votum to generate a distribution of model outputs rather than relying exclusively on a single deterministic forward pass.

# Data

You provide the data.

Votum is not restricted to one specific electoral dataset.

You can train experiments using combinations of:

- polling data;
- previous election results;
- municipal electoral data;
- demographic information;
- socioeconomic indicators;
- geographic information;
- candidate-related features;
- economic time series;
- other signals you consider relevant.

The default implementation expects three files:

``data/
├── municipios_final.csv
├── edge_index.csv
└── series_uf.csv``

### "municipios_final.csv"

Contains municipal-level information used by the spatial GNN.

The reference configuration includes features such as:

population
male/female population
area
electoral density
average income
literacy
HDI
2018 election results
2022 election results
latitude / longitude

These features can be changed in "Config.gnn_feature_cols".

### "edge_index.csv"

Defines the graph connecting municipalities.

Expected structure:

origem,destino
0,1
0,2
1,4
...

The reference dataset uses geographic k-nearest-neighbor relationships.

You may generate a completely different graph if desired.

### "series_uf.csv"

Contains temporal information for each state.

The default configuration supports signals such as:

date
polling intention by candidate
monthly inflation
12-month inflation
interest rate
poll availability

You can add, remove, or replace these features by changing:

Config.series_uf_feature_cols

This means Votum can be trained with polling data, historical election data, or a combination of multiple electoral signals.

# Configuration

Most important model and training parameters are exposed through "Config" or CLI arguments.

Example:

``python train_local.py \
  --data-dir ./data \
  --checkpoint-dir ./checkpoints \
  --epochs 20 \
  --batch-size 32 \
  --lr 2e-4 \
  --seq-len 128 \
  --hidden-transformer 1024 \
  --n-layers-transformer 24 \
  --hidden-gnn 256``

You are encouraged to change these values according to your hardware and experiment.

For example, a smaller configuration could use:

``python train_local.py \
  --hidden-transformer 256 \
  --n-layers-transformer 6 \
  --hidden-gnn 128 \
  --batch-size 8``

# Hardware

Votum supports:

``NVIDIA CUDA
Apple Silicon / MPS
CPU``

The device is automatically detected unless explicitly specified:

``python train_local.py --device cuda``

or:

``python train_local.py --device mps``

or:

``python train_local.py --device cpu``

Mixed precision ("fp16" / "bf16") is available when running on CUDA.

Large default configurations can be computationally expensive. If you are running without a powerful GPU, reduce the Transformer size, number of layers, sequence length, or batch size.

# Temporal Split

Election data is a time series.

For this reason, Votum uses a temporal train/validation split instead of a random split.

For example:

``Past                         Future
───────────────────┬────────────────────
      TRAIN         │     VALIDATION
───────────────────┴────────────────────
                   cutoff``

The cutoff can be changed with:

``--data-corte-treino-val 2026-07-01``

This is important because randomly mixing future and historical electoral observations could leak future information into training.

# Historical Targets

The reference implementation can construct training targets from previous election results.

However, historical candidates and current candidates are not necessarily identical.

For that reason, users should adapt the target construction to their own dataset and experiment.

Polling data can also be incorporated into the temporal features and is particularly useful when modeling candidates without directly comparable historical presidential results.

# Checkpoints

Training checkpoints are stored in:

``./checkpoints/``

The training configuration is also saved as:

``checkpoints/config.json``

allowing inference code to reconstruct the architecture used during training.

Training can be resumed using:

``python train_local.py \
  --resume-from ./checkpoints/step_500.pt``

# Customizing Votum

Votum is designed to be modified.

You can change:

✓ Candidates
✓ Polling sources
✓ Historical elections
✓ Municipal features
✓ Economic indicators
✓ Graph construction
✓ Temporal features
✓ Model dimensions
✓ Number of Transformer layers
✓ GNN architecture
✓ Training cutoff
✓ Training targets
✓ Sequence length

The included configuration represents one experiment, not the only way to use the architecture.

Researchers and developers are encouraged to adapt it to their own datasets and hypotheses.

# Limitations

Election modeling is inherently uncertain.

Predictions may be affected by:

- polling error;
- incomplete or biased datasets;
- turnout changes;
- late campaign events;
- distribution shifts;
- missing candidates or features;
- differences between historical and current elections;
- overfitting
- preprocessing choices;
- model configuration.

Votum does not "know" an election result in advance.

It estimates patterns from the information supplied to it.

# Responsible Use

When publishing results generated with Votum, we recommend clearly reporting:

Training data
Data cutoff
Polling sources
Model configuration
Checkpoint
Evaluation methodology
Date of inference

This makes experiments easier to reproduce and makes clear what information was available to the model at prediction time.

# License

Apache 2.0

---

Votum
Built by SynastrIA Networks.
