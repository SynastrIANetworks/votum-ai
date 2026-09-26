"""
Treino do modelo de previsão eleitoral (GNN espacial + Transformer temporal
+ head bayesiana), versão para rodar localmente em seu dispositivo.

É adaptada pra CUDA, MPS, & CPU, mas é preciso alterar as configurações, dependendo do hardware.

DADOS DE ENTRADA ESPERADOS (por padrão em ./data, ajustável com --data-dir):
  municipios_final.csv   -- 1 linha por município (5.570 linhas)
      colunas: cd_mun, nm_mun, cd_uf, nm_uf, populacao, pop_homens,
      pop_mulheres, area_km2, densidade_eleitoral_2022, renda_media_R$,
      taxa_alfabetizacao_15mais, votos_pt_2018, votos_pl_ou_psl_2018,
      votos_total_2018, votos_pt_2022, votos_pl_2022, votos_total_2022,
      latitude, longitude, idhm_2010, votos_total_2022_corrigido

  edge_index.csv         -- arestas do grafo espacial (k-NN geográfico)
      colunas: origem, destino (índices posicionais em municipios_final.csv)

  series_uf.csv          -- série temporal por UF-data
      colunas esperadas: cd_uf, data, <pesquisas por candidato>, ipca_mensal,
      selic_meta, ... (ver Config.series_uf_feature_cols -- ajustar aos
      nomes reais das colunas quando esse arquivo for gerado/validado)

NOTA SOBRE O QUE FICOU DE FORA:
  - Tokens de tipo de evento (ex: caso Toffoli/Renan Santos): não incluídos
    (esquema foi discutido internamente, mas nunca chegou a ser anotado.)
 
LIMITAÇÃO CONHECIDA DO ALVO DE TREINO (ver montar_alvo_historico):
  O único dado histórico de voto real que existe (2018/2022) só tem PT e PL
  -- os outros 4 candidatos (Caiado, Zema, Renan Santos, Cury) não
  concorreram à presidência nessas eleições sob esse rótulo. Por isso, recomendamos usar
  dados de pesquisas também.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields

# ==========================================================================
# 1. Candidatos (fechado: os 6 que participam dos debates)
# ==========================================================================

CANDIDATOS = [
    "Lula da Silva (PT)",
    "Flávio Bolsonaro (PL)",
    "Ronaldo Caiado (PSD)",
    "Romeu Zema (Novo)",
    "Renan Santos (Missão)",
    "Augusto Cury (Avante)",
]
CANDIDATO_TO_IDX = {nome: i for i, nome in enumerate(CANDIDATOS)}


# ==========================================================================
# 2. Config
# ==========================================================================

@dataclass
class Config:
    # --- Dimensões ---
    # seq_len = 128: o dado real disponível é mensal e cobre 67 meses
    # (2021-2026) -- um seq_len maior gera zero-padding sem ganho de sinal.
    # 128 dá folga pra crescer se granularidade semanal/diária for
    # adicionada depois.
    seq_len: int = 128
    batch_size: int = 32
    hidden_transformer: int = 1024
    n_layers_transformer: int = 24

    n_candidatos: int = len(CANDIDATOS)

    # --- GNN espacial ---
    n_nos: int = 5570
    hidden_gnn: int = 256
    n_camadas_gnn: int = 3
    gnn_feature_cols: list = field(default_factory=lambda: [
        "populacao", "pop_homens", "pop_mulheres", "area_km2",
        "densidade_eleitoral_2022", "renda_media_R$",
        "taxa_alfabetizacao_15mais", "idhm_2010",
        "votos_pt_2018", "votos_pl_ou_psl_2018", "votos_total_2018",
        "votos_pt_2022", "votos_pl_2022", "votos_total_2022_corrigido",
    ])

    # --- Transformer temporal (série por UF) ---
    series_uf_feature_cols: list = field(default_factory=lambda: [
        "ipca_mensal", "ipca_12m", "selic_meta", "tem_pesquisa",
        "intencao_Lula da Silva (PT)", "intencao_Flávio Bolsonaro (PL)",
        "intencao_Ronaldo Caiado (PSD)", "intencao_Romeu Zema (Novo)",
        "intencao_Renan Santos (Missão)", "intencao_Augusto Cury (Avante)",
    ])

    # --- Treino ---
    lr: float = 2e-4
    warmup_steps: int = 500
    n_epochs: int = 20
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    mixed_precision: str = "fp16"  # "fp16" | "bf16" | "no"

    log_every_n_steps: int = 50
    save_every_n_steps: int = 500
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    resume_from: str = ""
    num_workers: int = 0
    device: str = ""  # vazio = auto-detect

    # --- Split temporal (NUNCA aleatório em série temporal eleitoral) ---
    data_corte_treino_val: str = "2026-07-01"

    @property
    def gnn_in_dim(self):
        return len(self.gnn_feature_cols)

    @property
    def temporal_in_dim(self):
        return len(self.series_uf_feature_cols)


# ==========================================================================
# 3. Device / mixed precision helpers
# ==========================================================================

def get_device(preferred: str = ""):
    import torch

    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_amp_settings(device, mixed_precision: str):
    """
    Mixed precision só faz sentido em CUDA aqui (autocast+GradScaler). Em
    MPS/CPU, desliga -- evita erros silenciosos de precisão/instabilidade
    numérica que não valem a pena fora de GPU NVIDIA.
    """
    import torch

    if device.type == "cuda" and mixed_precision in ("fp16", "bf16"):
        dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
        use_scaler = dtype == torch.float16
        return True, dtype, use_scaler
    return False, torch.float32, False


# ==========================================================================
# 4. Modelo
# ==========================================================================

def build_model(cfg: Config):
    import torch
    import torch.nn as nn
    from torch_geometric.nn import SAGEConv, GATv2Conv

    class GNNEspacial(nn.Module):
        def __init__(self, in_dim, hidden_dim, n_camadas):
            super().__init__()
            self.sage = SAGEConv(in_dim, hidden_dim)
            self.gat_layers = nn.ModuleList([
                GATv2Conv(hidden_dim, hidden_dim) for _ in range(n_camadas - 1)
            ])

        def forward(self, x, edge_index):
            x = self.sage(x, edge_index).relu()
            for gat in self.gat_layers:
                x = gat(x, edge_index).relu()
            return x  # [n_nos, hidden_dim]

    class TransformerTemporal(nn.Module):
        def __init__(self, dim_entrada, hidden_dim, n_layers, seq_len):
            super().__init__()
            self.proj_entrada = nn.Linear(dim_entrada, hidden_dim)
            self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, batch_first=True,
                dim_feedforward=hidden_dim * 4,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        def forward(self, x):
            seq_len_atual = x.shape[1]
            x = self.proj_entrada(x) + self.pos_embedding[:, :seq_len_atual, :]
            return self.encoder(x)

    class CrossAttention(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)

        def forward(self, temporal_out, espacial_out_por_uf):
            espacial_expandido = espacial_out_por_uf.unsqueeze(1)
            saida, _ = self.attn(temporal_out, espacial_expandido, espacial_expandido)
            return saida

    class HeadBayesiana(nn.Module):
        def __init__(self, hidden_dim, n_candidatos):
            super().__init__()
            self.dropout = nn.Dropout(p=0.3)
            self.fc = nn.Linear(hidden_dim, n_candidatos)

        def forward(self, x, mc_samples=1):
            saidas = []
            for _ in range(mc_samples):
                h = self.dropout(x)
                saidas.append(torch.softmax(self.fc(h), dim=-1))
            return torch.stack(saidas)  # [mc_samples, batch, n_candidatos]

    class ModeloCompleto(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.gnn = GNNEspacial(cfg.gnn_in_dim, cfg.hidden_gnn, cfg.n_camadas_gnn)
            self.temporal = TransformerTemporal(
                dim_entrada=cfg.temporal_in_dim,
                hidden_dim=cfg.hidden_transformer,
                n_layers=cfg.n_layers_transformer,
                seq_len=cfg.seq_len,
            )
            self.cross_attn = CrossAttention(cfg.hidden_transformer)
            self.proj_gnn_to_temporal = nn.Linear(cfg.hidden_gnn, cfg.hidden_transformer)
            self.head = HeadBayesiana(cfg.hidden_transformer, n_candidatos=cfg.n_candidatos)

        def forward(self, x_temporal, x_gnn, edge_index, uf_idx_batch, mc_samples=1):
            """
            x_temporal: [batch, seq_len, temporal_in_dim]
            x_gnn: [n_nos, gnn_in_dim] -- grafo completo (não batcheado, é pequeno)
            edge_index: [2, n_edges]
            uf_idx_batch: [batch] -- índice do NÓ representativo da UF de cada
                          amostra (aproximação: primeiro município encontrado
                          daquela UF -- ver DatasetEleitoral.uf_to_no_idx)
            """
            espacial_out = self.gnn(x_gnn, edge_index)
            espacial_out = self.proj_gnn_to_temporal(espacial_out)
            espacial_out_batch = espacial_out[uf_idx_batch]

            temporal_out = self.temporal(x_temporal)
            combinado = self.cross_attn(temporal_out, espacial_out_batch)
            pooled = combinado.mean(dim=1)
            return self.head(pooled, mc_samples=mc_samples)

    return ModeloCompleto(cfg)


# ==========================================================================
# 5. Carregamento de dados reais
# ==========================================================================

def carregar_municipios(cfg: Config, path_csv: str):
    """Lê municipios_final.csv e monta o tensor de features da GNN (normalizado)."""
    import torch

    if not os.path.exists(path_csv):
        raise FileNotFoundError(
            f"Não encontrei {path_csv}. Confira --data-dir e se o arquivo "
            "municipios_final.csv está lá."
        )

    with open(path_csv, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))

    assert len(linhas) == cfg.n_nos, (
        f"Esperado {cfg.n_nos} municípios, encontrado {len(linhas)}. "
        "Confira se o CSV é o municipal correto (não o agregado por UF)."
    )

    features = []
    cd_uf_por_no = []
    faltantes = 0

    for linha in linhas:
        vetor = []
        for col in cfg.gnn_feature_cols:
            val = linha.get(col, "")
            if val in ("", None):
                faltantes += 1
                val = 0.0
            vetor.append(float(val))
        features.append(vetor)
        cd_uf_por_no.append(linha["cd_uf"])

    x_gnn = torch.tensor(features, dtype=torch.float32)

    # Normalização por coluna -- essencial, população e IDH têm escalas
    # muito diferentes (milhões vs. 0-1) e isso quebra o treino sem isso
    media = x_gnn.mean(dim=0, keepdim=True)
    desvio = x_gnn.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_gnn = (x_gnn - media) / desvio

    if faltantes > 0:
        print(f"[aviso] {faltantes} valores de feature faltantes preenchidos com 0 "
              f"(idhm_2010 tem 5 municípios sem valor -- vira a média pós-normalização)")

    return x_gnn, cd_uf_por_no


def carregar_edge_index(path_csv: str):
    import torch

    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"Não encontrei {path_csv}. Confira --data-dir.")

    with open(path_csv, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    origem = [int(r["origem"]) for r in linhas]
    destino = [int(r["destino"]) for r in linhas]
    return torch.tensor([origem, destino], dtype=torch.long)


def carregar_series_uf(cfg: Config, path_csv: str):
    """
    Lê series_uf.csv. Espera colunas: cd_uf, data, + cfg.series_uf_feature_cols.
    Retorna dict: cd_uf -> lista de (data, vetor_features) ordenada por data.
    """
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"Não encontrei {path_csv}. Confira --data-dir.")

    with open(path_csv, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))

    por_uf = defaultdict(list)
    faltando_cols = set()
    for r in linhas:
        vetor = []
        for col in cfg.series_uf_feature_cols:
            if col not in r:
                faltando_cols.add(col)
            vetor.append(float(r.get(col, 0.0) or 0.0))
        por_uf[r["cd_uf"]].append((r["data"], vetor))

    if faltando_cols:
        print(f"[aviso] colunas ausentes em series_uf.csv (preenchidas com 0): "
              f"{sorted(faltando_cols)}")

    for uf in por_uf:
        por_uf[uf].sort(key=lambda t: t[0])

    return por_uf


def construir_uf_to_no_idx(cd_uf_por_no):
    """Mapeia cd_uf -> índice do primeiro município (nó da GNN) daquela UF."""
    uf_to_no_idx = {}
    for i, cd_uf in enumerate(cd_uf_por_no):
        if cd_uf not in uf_to_no_idx:
            uf_to_no_idx[cd_uf] = i
    return uf_to_no_idx


class DatasetEleitoral:
    """
    Cada amostra é uma janela temporal (seq_len passos) de UMA UF, ligada
    ao nó da GNN representativo daquela UF.

    Split temporal (não aleatório): tudo antes de data_corte_treino_val vira
    treino, o resto vira validação -- crítico em série temporal eleitoral,
    onde um split aleatório vazaria informação do futuro para o passado.
    """

    def __init__(self, cfg: Config, cd_uf_por_no, series_por_uf, split="treino"):
        self.cfg = cfg
        self.series_por_uf = series_por_uf

        self.uf_to_no_idx = construir_uf_to_no_idx(cd_uf_por_no)

        self.amostras = []
        cd_uf_sem_no = set()
        for cd_uf, serie in series_por_uf.items():
            if cd_uf not in self.uf_to_no_idx:
                cd_uf_sem_no.add(cd_uf)
                continue
            datas = [d for d, _ in serie]
            if split == "treino":
                pontos_validos = [i for i, d in enumerate(datas) if d < cfg.data_corte_treino_val]
            else:
                pontos_validos = [i for i, d in enumerate(datas) if d >= cfg.data_corte_treino_val]

            for idx_fim in pontos_validos:
                self.amostras.append((cd_uf, idx_fim))

        if cd_uf_sem_no:
            print(f"[aviso] cd_uf em series_uf.csv sem município correspondente "
                  f"(ignorados): {sorted(cd_uf_sem_no)}")

    def __len__(self):
        return len(self.amostras)

    def __getitem__(self, idx):
        import torch

        cd_uf, idx_fim = self.amostras[idx]
        serie = self.series_por_uf[cd_uf]
        idx_inicio = max(0, idx_fim + 1 - self.cfg.seq_len)
        janela = serie[idx_inicio: idx_fim + 1]

        x_temporal = torch.tensor([v for _, v in janela], dtype=torch.float32)
        if x_temporal.shape[0] < self.cfg.seq_len:
            pad = torch.zeros(self.cfg.seq_len - x_temporal.shape[0], x_temporal.shape[1])
            x_temporal = torch.cat([pad, x_temporal], dim=0)

        return {
            "x_temporal": x_temporal,
            "uf_idx": self.uf_to_no_idx[cd_uf],
        }


def montar_alvo_historico(cfg: Config, linha_municipio: dict):
    """
    Alvo de treino a partir do único dado histórico real disponível: votos
    PT vs PL em 2022 no município representativo da UF. Os outros 4
    candidatos (sem histórico presidencial sob esse rótulo) recebem uma
    probabilidade residual pequena em vez de zero, pra não travar o KLDiv.
    """
    import torch

    votos_pt = float(linha_municipio["votos_pt_2022"])
    votos_pl = float(linha_municipio["votos_pl_2022"])
    total = votos_pt + votos_pl

    dist = torch.full((cfg.n_candidatos,), 1e-3)
    if total > 0:
        dist[CANDIDATO_TO_IDX["Lula da Silva (PT)"]] = votos_pt / total
        dist[CANDIDATO_TO_IDX["Flávio Bolsonaro (PL)"]] = votos_pl / total
    else:
        dist[:] = 1.0 / cfg.n_candidatos
    return dist / dist.sum()


def montar_collate_fn(x_gnn_raw_linhas, cfg):
    import torch

    def collate_fn(batch):
        x_temporal = torch.stack([b["x_temporal"] for b in batch])
        uf_idx = torch.tensor([b["uf_idx"] for b in batch], dtype=torch.long)
        y_dist = torch.stack([
            montar_alvo_historico(cfg, x_gnn_raw_linhas[u.item()]) for u in uf_idx
        ])
        return {"x_temporal": x_temporal, "uf_idx": uf_idx, "y_dist": y_dist}

    return collate_fn


# ==========================================================================
# 6. Loop de treino / validação
# ==========================================================================

def evaluate(cfg: Config, model, dataloader, x_gnn, edge_index, device, loss_fn):
    import torch

    if dataloader is None or len(dataloader.dataset) == 0:
        return None

    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            x_temporal = batch["x_temporal"].to(device)
            uf_idx = batch["uf_idx"].to(device)
            y_true_dist = batch["y_dist"].to(device)

            pred = model(x_temporal, x_gnn, edge_index, uf_idx, mc_samples=1).squeeze(0)
            log_pred = torch.log(pred.clamp(min=1e-8))
            loss = loss_fn(log_pred, y_true_dist)

            total_loss += loss.item()
            n_batches += 1

    model.train()
    return total_loss / max(1, n_batches)


def train_loop(cfg: Config, model, dataloader_treino, dataloader_val, x_gnn, edge_index, device):
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR

    model.to(device)
    x_gnn = x_gnn.to(device)
    edge_index = edge_index.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(1, cfg.n_epochs * len(dataloader_treino))
    scheduler = OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=total_steps,
        pct_start=min(0.3, cfg.warmup_steps / total_steps),
    )

    use_amp, amp_dtype, use_scaler = get_amp_settings(device, cfg.mixed_precision)
    scaler = torch.amp.GradScaler(device.type if device.type == "cuda" else "cpu", enabled=use_scaler)
    loss_fn = nn.KLDivLoss(reduction="batchmean")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    start_epoch = 0
    step = 0
    best_val_loss = float("inf")

    if cfg.resume_from:
        if not os.path.exists(cfg.resume_from):
            raise FileNotFoundError(f"Checkpoint de resume não encontrado: {cfg.resume_from}")
        ckpt = torch.load(cfg.resume_from, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt.get("optimizer_state_dict", optimizer.state_dict()))
            start_epoch = ckpt.get("epoch", 0)
            step = ckpt.get("step", 0)
            best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        else:
            model.load_state_dict(ckpt)  # checkpoint antigo, só pesos
        print(f"[resume] retomando de {cfg.resume_from} (epoch={start_epoch}, step={step})")

    for epoch in range(start_epoch, cfg.n_epochs):
        model.train()
        for batch in dataloader_treino:
            x_temporal = batch["x_temporal"].to(device)
            uf_idx = batch["uf_idx"].to(device)
            y_true_dist = batch["y_dist"].to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred = model(x_temporal, x_gnn, edge_index, uf_idx, mc_samples=1).squeeze(0)
                log_pred = torch.log(pred.clamp(min=1e-8))
                loss = loss_fn(log_pred, y_true_dist)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            if step % cfg.log_every_n_steps == 0:
                print(f"epoch {epoch} | step {step} | loss {loss.item():.4f} | "
                      f"lr {scheduler.get_last_lr()[0]:.2e}")

            if step % cfg.save_every_n_steps == 0:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "step": step,
                    "best_val_loss": best_val_loss,
                }, f"{cfg.checkpoint_dir}/step_{step}.pt")

        val_loss = evaluate(cfg, model, dataloader_val, x_gnn, edge_index, device, loss_fn)
        if val_loss is not None:
            print(f"[val] epoch {epoch} | loss {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{cfg.checkpoint_dir}/best.pt")
                print(f"[val] novo melhor checkpoint salvo (loss {best_val_loss:.4f})")
        else:
            print(f"[val] epoch {epoch} | sem amostras de validação "
                  f"(confira data_corte_treino_val)")

    return model


# ==========================================================================
# 7. Execução local
# ==========================================================================

def treinar(cfg: Config):
    import torch
    from torch.utils.data import DataLoader

    device = get_device(cfg.device)
    print(f"Rodando em: {device} | seq_len={cfg.seq_len} | batch_size={cfg.batch_size} | "
          f"hidden_transformer={cfg.hidden_transformer} | n_layers_transformer={cfg.n_layers_transformer}")

    # Salva a config junto com os checkpoints -- o predict_local.py precisa
    # dela pra recriar a arquitetura exata do modelo treinado.
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.checkpoint_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    if device.type != "cuda":
        print("[aviso] sem CUDA disponível -- mixed precision desligado e o treino "
              "tende a ser bem mais lento. Considere reduzir --hidden-transformer, "
              "--n-layers-transformer e --batch-size pra rodar num tamanho razoável.")

    path_municipios = os.path.join(cfg.data_dir, "municipios_final.csv")
    path_edges = os.path.join(cfg.data_dir, "edge_index.csv")
    path_series = os.path.join(cfg.data_dir, "series_uf.csv")

    x_gnn, cd_uf_por_no = carregar_municipios(cfg, path_municipios)
    edge_index = carregar_edge_index(path_edges)
    series_por_uf = carregar_series_uf(cfg, path_series)

    with open(path_municipios, encoding="utf-8") as f:
        x_gnn_raw_linhas = list(csv.DictReader(f))

    dataset_treino = DatasetEleitoral(cfg, cd_uf_por_no, series_por_uf, split="treino")
    dataset_val = DatasetEleitoral(cfg, cd_uf_por_no, series_por_uf, split="val")
    print(f"Amostras de treino: {len(dataset_treino)} | validação: {len(dataset_val)}")
    if len(dataset_treino) == 0:
        raise RuntimeError(
            "Dataset de treino vazio. Confira se series_uf.csv tem datas "
            f"anteriores a {cfg.data_corte_treino_val} e se os cd_uf batem "
            "com os do arquivo municipal."
        )

    collate_fn = montar_collate_fn(x_gnn_raw_linhas, cfg)
    dataloader_treino = DataLoader(
        dataset_treino, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=cfg.num_workers,
    )
    dataloader_val = None
    if len(dataset_val) > 0:
        dataloader_val = DataLoader(
            dataset_val, batch_size=cfg.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=cfg.num_workers,
        )

    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros do modelo: {n_params:,}")

    model = train_loop(cfg, model, dataloader_treino, dataloader_val, x_gnn, edge_index, device)

    torch.save(model.state_dict(), f"{cfg.checkpoint_dir}/final.pt")
    print("Treino concluído. Checkpoint salvo em", cfg.checkpoint_dir)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="./data", help="Pasta com municipios_final.csv, edge_index.csv, series_uf.csv")
    p.add_argument("--checkpoint-dir", default="./checkpoints")
    p.add_argument("--resume-from", default="", help="Caminho de um checkpoint .pt pra retomar o treino")
    p.add_argument("--device", default="", help="cuda | mps | cpu (vazio = auto-detect)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--hidden-transformer", type=int, default=1024)
    p.add_argument("--n-layers-transformer", type=int, default=24)
    p.add_argument("--hidden-gnn", type=int, default=256)
    p.add_argument("--mixed-precision", default="fp16", choices=["fp16", "bf16", "no"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every-n-steps", type=int, default=50)
    p.add_argument("--save-every-n-steps", type=int, default=500)
    p.add_argument("--data-corte-treino-val", default="2026-07-01")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        device=args.device,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seq_len=args.seq_len,
        hidden_transformer=args.hidden_transformer,
        n_layers_transformer=args.n_layers_transformer,
        hidden_gnn=args.hidden_gnn,
        mixed_precision=args.mixed_precision,
        num_workers=args.num_workers,
        log_every_n_steps=args.log_every_n_steps,
        save_every_n_steps=args.save_every_n_steps,
        data_corte_treino_val=args.data_corte_treino_val,
    )
    treinar(cfg)


if __name__ == "__main__":
    main()
