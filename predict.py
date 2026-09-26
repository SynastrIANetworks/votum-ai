"""
Gera as previsões do modelo treinado (train_local.py) por UF e um agregado
nacional ponderado por população.

Reaproveita as funções de carregamento de dados e a definição do modelo de
train_local.py -- rode este script na mesma pasta que ele.

Incerteza (desvio padrão por candidato) vem de MC-Dropout: a HeadBayesiana
já foi desenhada pra isso (múltiplas passadas com dropout ativo só na head;
GNN e Transformer ficam em modo determinístico/eval).

COMO RODAR:
  python predict_local.py --checkpoint ./checkpoints/best.pt --data-dir ./data

  Se ./checkpoints/config.json existir (gerado automaticamente pelo
  train_local.py atual), a arquitetura é recriada sozinha. Se o checkpoint
  foi treinado com uma versão anterior do script (sem config.json), passe
  os mesmos overrides de arquitetura usados no treino
  (--hidden-transformer, --n-layers-transformer, --hidden-gnn, --seq-len).
"""

import argparse
import csv
import json
import os
from dataclasses import replace

from train_local import (
    CANDIDATOS,
    Config,
    build_model,
    carregar_edge_index,
    carregar_municipios,
    carregar_series_uf,
    construir_uf_to_no_idx,
    get_device,
)


def carregar_config(checkpoint_path: str, config_arg: str, cli_overrides: dict) -> Config:
    """
    Prioridade: --config explícito > config.json ao lado do checkpoint >
    defaults do Config combinados com os overrides de arquitetura passados
    na linha de comando (fallback pra checkpoints antigos sem config.json).
    """
    candidatos_config = []
    if config_arg:
        candidatos_config.append(config_arg)
    candidatos_config.append(os.path.join(os.path.dirname(checkpoint_path), "config.json"))

    for path in candidatos_config:
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                dados = json.load(f)
            campos_validos = {c.name for c in Config.__dataclass_fields__.values()}
            dados_filtrados = {k: v for k, v in dados.items() if k in campos_validos}
            print(f"[config] carregada de {path}")
            return Config(**dados_filtrados)

    print("[aviso] config.json não encontrado -- usando defaults + overrides da CLI. "
          "Se o checkpoint foi treinado com hiperparâmetros diferentes, os pesos "
          "não vão carregar (erro de shape).")
    return Config(**cli_overrides)


def carregar_pesos_modelo(model, checkpoint_path, device):
    import torch

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)  # final.pt / best.pt salvam só os pesos
    return model


def montar_janela_mais_recente(cfg: Config, serie):
    """
    Mesma lógica de janela/padding do DatasetEleitoral, mas sempre pegando
    os últimos seq_len pontos disponíveis (a janela mais recente possível),
    já que aqui queremos prever o "próximo" resultado, não reproduzir splits
    de treino/validação.
    """
    import torch

    janela = serie[-cfg.seq_len:]
    x = torch.tensor([v for _, v in janela], dtype=torch.float32)
    if x.shape[0] < cfg.seq_len:
        pad = torch.zeros(cfg.seq_len - x.shape[0], x.shape[1])
        x = torch.cat([pad, x], dim=0)
    return x


def calcular_pesos_populacao_por_uf(x_gnn_raw_linhas):
    pesos = {}
    for linha in x_gnn_raw_linhas:
        cd_uf = linha["cd_uf"]
        pesos[cd_uf] = pesos.get(cd_uf, 0.0) + float(linha.get("populacao", 0.0) or 0.0)
    return pesos


def prever(args):
    import torch

    device = get_device(args.device)

    overrides = {
        "seq_len": args.seq_len,
        "hidden_transformer": args.hidden_transformer,
        "n_layers_transformer": args.n_layers_transformer,
        "hidden_gnn": args.hidden_gnn,
        "data_dir": args.data_dir,
    }
    cfg = carregar_config(args.checkpoint, args.config, overrides)
    cfg = replace(cfg, data_dir=args.data_dir)  # sempre respeita --data-dir da CLI

    print(f"Rodando em: {device}")

    path_municipios = os.path.join(cfg.data_dir, "municipios_final.csv")
    path_edges = os.path.join(cfg.data_dir, "edge_index.csv")
    path_series = os.path.join(cfg.data_dir, "series_uf.csv")

    x_gnn, cd_uf_por_no = carregar_municipios(cfg, path_municipios)
    edge_index = carregar_edge_index(path_edges)
    series_por_uf = carregar_series_uf(cfg, path_series)

    with open(path_municipios, encoding="utf-8") as f:
        x_gnn_raw_linhas = list(csv.DictReader(f))

    uf_to_no_idx = construir_uf_to_no_idx(cd_uf_por_no)
    pesos_populacao = calcular_pesos_populacao_por_uf(x_gnn_raw_linhas)

    ufs_validas = [uf for uf in series_por_uf if uf in uf_to_no_idx]
    ufs_ignoradas = [uf for uf in series_por_uf if uf not in uf_to_no_idx]
    if ufs_ignoradas:
        print(f"[aviso] UFs em series_uf.csv sem município correspondente (ignoradas): "
              f"{sorted(ufs_ignoradas)}")
    if not ufs_validas:
        raise RuntimeError("Nenhuma UF válida encontrada -- confira series_uf.csv e municipios_final.csv.")

    model = build_model(cfg)
    model = carregar_pesos_modelo(model, args.checkpoint, device)
    model.to(device)
    x_gnn = x_gnn.to(device)
    edge_index = edge_index.to(device)

    # GNN e Transformer em modo determinístico; só a dropout da head fica
    # ativa (é o mecanismo de incerteza intencional do modelo -- ver
    # HeadBayesiana em train_local.py).
    model.eval()
    model.head.dropout.train()

    x_temporal = torch.stack([
        montar_janela_mais_recente(cfg, series_por_uf[uf]) for uf in ufs_validas
    ]).to(device)
    uf_idx = torch.tensor([uf_to_no_idx[uf] for uf in ufs_validas], dtype=torch.long).to(device)

    with torch.no_grad():
        amostras = model(x_temporal, x_gnn, edge_index, uf_idx, mc_samples=args.mc_samples)
        # amostras: [mc_samples, n_ufs, n_candidatos]
        media = amostras.mean(dim=0).cpu()
        desvio = amostras.std(dim=0).cpu()

    linhas_saida = []
    for i, uf in enumerate(ufs_validas):
        for c, candidato in enumerate(CANDIDATOS):
            linhas_saida.append({
                "uf": uf,
                "candidato": candidato,
                "probabilidade_media": media[i, c].item(),
                "probabilidade_std": desvio[i, c].item(),
            })

    # Agregado nacional: média por UF ponderada pela população total da UF
    # (soma da coluna populacao dos municípios daquela UF).
    pesos = torch.tensor([pesos_populacao.get(uf, 0.0) for uf in ufs_validas])
    if pesos.sum() > 0:
        pesos_norm = pesos / pesos.sum()
        media_nacional = (media * pesos_norm.unsqueeze(1)).sum(dim=0)
        for c, candidato in enumerate(CANDIDATOS):
            linhas_saida.append({
                "uf": "BR (agregado, ponderado por população)",
                "candidato": candidato,
                "probabilidade_media": media_nacional[c].item(),
                "probabilidade_std": "",
            })
    else:
        media_nacional = None
        print("[aviso] soma de população zerada -- pulando agregado nacional")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["uf", "candidato", "probabilidade_media", "probabilidade_std"])
        writer.writeheader()
        writer.writerows(linhas_saida)
    print(f"Previsões salvas em {args.output}")

    if media_nacional is not None:
        ranking = sorted(zip(CANDIDATOS, media_nacional.tolist()), key=lambda t: -t[1])
        print("\nRanking nacional (agregado ponderado por população):")
        for candidato, prob in ranking:
            print(f"  {candidato:35s} {prob * 100:5.1f}%")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Caminho do checkpoint .pt (ex: ./checkpoints/best.pt)")
    p.add_argument("--config", default="", help="Caminho de um config.json explícito (opcional)")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--output", default="previsao_resultados.csv")
    p.add_argument("--device", default="", help="cuda | mps | cpu (vazio = auto-detect)")
    p.add_argument("--mc-samples", type=int, default=30, help="Nº de passadas MC-Dropout pra estimar incerteza")
    # Fallback de arquitetura pra checkpoints sem config.json salvo:
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--hidden-transformer", type=int, default=1024)
    p.add_argument("--n-layers-transformer", type=int, default=24)
    p.add_argument("--hidden-gnn", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    prever(args)


if __name__ == "__main__":
    main()
