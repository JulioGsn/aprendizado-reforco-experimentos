#!/usr/bin/env python3
"""Executa comparacoes reproduziveis entre Q-learning e SARSA."""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AMBIENTES = ["Blackjack-v1", "CliffWalking-v1", "FrozenLake-v1", "Taxi-v4"]
ALGORITMOS = ["q_learning", "sarsa"]


@dataclass(frozen=True)
class Configuracao:
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_inicial: float = 1.0
    epsilon_final: float = 0.05
    episodios_avaliacao: int = 500
    limite_passos: int = 1_000


EPISODIOS = {
    "teste": {
        "Blackjack-v1": 300,
        "CliffWalking-v1": 200,
        "FrozenLake-v1": 500,
        "Taxi-v4": 300,
    },
    "relatorio": {
        "Blackjack-v1": 50_000,
        "CliffWalking-v1": 5_000,
        "FrozenLake-v1": 30_000,
        "Taxi-v4": 10_000,
    },
}


def chave_estado(observacao):
    if isinstance(observacao, np.ndarray):
        return tuple(observacao.tolist())
    if isinstance(observacao, list):
        return tuple(observacao)
    return observacao


def escolher_acao(q, estado, numero_acoes, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(numero_acoes))

    valores = q[estado]
    melhores = np.flatnonzero(valores == valores.max())
    return int(rng.choice(melhores))


def epsilon_no_episodio(config, episodio, total_episodios):
    progresso = episodio / max(total_episodios - 1, 1)
    return config.epsilon_inicial + progresso * (
        config.epsilon_final - config.epsilon_inicial
    )


def episodio_bem_sucedido(ambiente_id, recompensa_total, terminated):
    if ambiente_id == "Blackjack-v1":
        return recompensa_total > 0
    return terminated


def treinar(ambiente_id, algoritmo, semente, episodios, config):
    env = gym.make(ambiente_id)
    numero_acoes = env.action_space.n
    q = defaultdict(lambda: np.zeros(numero_acoes, dtype=np.float64))
    rng = np.random.default_rng(semente)
    registros = []
    inicio = time.perf_counter()

    for episodio in range(episodios):
        estado, _ = env.reset(seed=semente + episodio)
        estado = chave_estado(estado)
        epsilon = epsilon_no_episodio(config, episodio, episodios)
        acao = escolher_acao(q, estado, numero_acoes, epsilon, rng)
        recompensa_total = 0.0
        passos = 0
        concluiu = False

        for _ in range(config.limite_passos):
            proximo_estado, recompensa, terminated, truncated, _ = env.step(acao)
            proximo_estado = chave_estado(proximo_estado)
            terminou = terminated or truncated
            recompensa_total += float(recompensa)
            passos += 1
            concluiu = terminated

            if algoritmo == "q_learning":
                futuro = 0.0 if terminou else float(q[proximo_estado].max())
                alvo = float(recompensa) + config.gamma * futuro
                q[estado][acao] += config.alpha * (alvo - q[estado][acao])
                proxima_acao = None
            else:
                proxima_acao = None
                futuro = 0.0
                if not terminou:
                    proxima_acao = escolher_acao(
                        q, proximo_estado, numero_acoes, epsilon, rng
                    )
                    futuro = float(q[proximo_estado][proxima_acao])
                alvo = float(recompensa) + config.gamma * futuro
                q[estado][acao] += config.alpha * (alvo - q[estado][acao])

            if terminou:
                break

            estado = proximo_estado
            if algoritmo == "q_learning":
                acao = escolher_acao(q, estado, numero_acoes, epsilon, rng)
            else:
                acao = proxima_acao

        registros.append(
            {
                "ambiente": ambiente_id,
                "algoritmo": algoritmo,
                "semente": semente,
                "episodio": episodio + 1,
                "epsilon": epsilon,
                "recompensa": recompensa_total,
                "passos": passos,
                "sucesso": int(
                    episodio_bem_sucedido(ambiente_id, recompensa_total, concluiu)
                ),
            }
        )

    duracao = time.perf_counter() - inicio
    env.close()
    return q, registros, duracao


def avaliar(ambiente_id, q, semente, episodios, limite_passos):
    env = gym.make(ambiente_id)
    rng = np.random.default_rng(semente + 1_000_000)
    recompensas = []
    passos_lista = []
    passos_sucesso = []
    sucessos = []

    for episodio in range(episodios):
        estado, _ = env.reset(seed=semente + 2_000_000 + episodio)
        estado = chave_estado(estado)
        recompensa_total = 0.0
        passos = 0
        concluiu = False

        for _ in range(limite_passos):
            valores = q[estado]
            melhores = np.flatnonzero(valores == valores.max())
            acao = int(rng.choice(melhores))
            proximo_estado, recompensa, terminated, truncated, _ = env.step(acao)
            estado = chave_estado(proximo_estado)
            recompensa_total += float(recompensa)
            passos += 1
            concluiu = terminated
            if terminated or truncated:
                break

        recompensas.append(recompensa_total)
        passos_lista.append(passos)
        sucesso = int(
            episodio_bem_sucedido(ambiente_id, recompensa_total, concluiu)
        )
        sucessos.append(sucesso)
        if sucesso:
            passos_sucesso.append(passos)

    env.close()
    return {
        "recompensa_media": float(np.mean(recompensas)),
        "recompensa_desvio_episodios": float(np.std(recompensas, ddof=1)),
        "passos_medios": float(np.mean(passos_lista)),
        "passos_medios_sucesso": (
            float(np.mean(passos_sucesso)) if passos_sucesso else float("nan")
        ),
        "taxa_sucesso": float(np.mean(sucessos)),
    }


def gerar_graficos(dados, diretorio):
    for ambiente, grupo_ambiente in dados.groupby("ambiente"):
        plt.figure(figsize=(10, 5.5))
        total_episodios = int(grupo_ambiente["episodio"].max())
        janela = max(10, total_episodios // 100)

        for algoritmo, grupo_algoritmo in grupo_ambiente.groupby("algoritmo"):
            por_episodio = grupo_algoritmo.groupby("episodio")["recompensa"]
            media = por_episodio.mean().rolling(janela, min_periods=1).mean()
            desvio = por_episodio.std().fillna(0).rolling(janela, min_periods=1).mean()
            x = media.index.to_numpy()
            y = media.to_numpy()
            d = desvio.to_numpy()
            rotulo = "Q-learning" if algoritmo == "q_learning" else "SARSA"
            plt.plot(x, y, label=rotulo, linewidth=1.8)
            plt.fill_between(x, y - d, y + d, alpha=0.15)

        plt.title(f"Curva de aprendizado - {ambiente}")
        plt.xlabel("Episódio")
        plt.ylabel(f"Recompensa média móvel (janela={janela})")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        nome = ambiente.lower().replace("-", "_")
        plt.savefig(diretorio / f"curva_{nome}.png", dpi=180)
        plt.close()


def executar(args):
    config = Configuracao(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_inicial=args.epsilon_inicial,
        epsilon_final=args.epsilon_final,
        episodios_avaliacao=(
            args.episodios_avaliacao
            if args.episodios_avaliacao is not None
            else (50 if args.modo == "teste" else 1_000)
        ),
        limite_passos=args.limite_passos,
    )
    total_sementes = (
        args.sementes
        if args.sementes is not None
        else (2 if args.modo == "teste" else 10)
    )
    sementes = list(range(total_sementes))
    episodios_por_ambiente = {
        ambiente: (
            args.episodios
            if args.episodios is not None
            else EPISODIOS[args.modo][ambiente]
        )
        for ambiente in args.ambientes
    }
    diretorio = Path(args.saida)
    diretorio.mkdir(parents=True, exist_ok=True)
    todos_episodios = []
    avaliacoes = []

    metadados = {
        "modo": args.modo,
        "ambientes": args.ambientes,
        "algoritmos": ALGORITMOS,
        "sementes": sementes,
        "episodios_treinamento": episodios_por_ambiente,
        "configuracao": asdict(config),
        "python": platform.python_version(),
        "gymnasium": gym.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (diretorio / "configuracao.json").write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for ambiente in args.ambientes:
        for algoritmo in ALGORITMOS:
            for semente in sementes:
                episodios = episodios_por_ambiente[ambiente]
                print(
                    f"Treinando {algoritmo} em {ambiente}, semente {semente}, "
                    f"{episodios} episodios...",
                    flush=True,
                )
                q, registros, duracao = treinar(
                    ambiente, algoritmo, semente, episodios, config
                )
                todos_episodios.extend(registros)
                resultado = avaliar(
                    ambiente,
                    q,
                    semente,
                    config.episodios_avaliacao,
                    config.limite_passos,
                )
                resultado.update(
                    {
                        "ambiente": ambiente,
                        "algoritmo": algoritmo,
                        "semente": semente,
                        "episodios_treinamento": episodios,
                        "episodios_avaliacao": config.episodios_avaliacao,
                        "duracao_segundos": duracao,
                        "estados_na_tabela_q": len(q),
                    }
                )
                avaliacoes.append(resultado)

    df_episodios = pd.DataFrame(todos_episodios)
    df_avaliacoes = pd.DataFrame(avaliacoes)
    df_episodios.to_csv(diretorio / "episodios.csv", index=False)
    df_avaliacoes.to_csv(diretorio / "avaliacao_por_semente.csv", index=False)

    resumo = (
        df_avaliacoes.groupby(["ambiente", "algoritmo"])
        .agg(
            recompensa_media=("recompensa_media", "mean"),
            recompensa_desvio_entre_sementes=("recompensa_media", "std"),
            passos_medios=("passos_medios", "mean"),
            passos_medios_sucesso=("passos_medios_sucesso", "mean"),
            taxa_sucesso_media=("taxa_sucesso", "mean"),
            duracao_media_segundos=("duracao_segundos", "mean"),
            estados_na_tabela_q_medios=("estados_na_tabela_q", "mean"),
        )
        .reset_index()
    )
    resumo.to_csv(diretorio / "resumo.csv", index=False)
    gerar_graficos(df_episodios, diretorio)

    print("\nResumo da avaliacao sem exploracao:")
    print(resumo.to_string(index=False))
    print(f"\nArquivos gravados em: {diretorio.resolve()}")


def argumentos():
    parser = argparse.ArgumentParser(
        description="Compara Q-learning e SARSA em ambientes Toy Text."
    )
    parser.add_argument("--modo", choices=["teste", "relatorio"], default="teste")
    parser.add_argument(
        "--ambientes",
        nargs="+",
        choices=AMBIENTES,
        default=AMBIENTES,
    )
    parser.add_argument("--saida", default="resultados_tabulares")
    parser.add_argument("--episodios", type=int)
    parser.add_argument("--sementes", type=int)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-inicial", type=float, default=1.0)
    parser.add_argument("--epsilon-final", type=float, default=0.05)
    parser.add_argument("--episodios-avaliacao", type=int)
    parser.add_argument("--limite-passos", type=int, default=1_000)
    return parser.parse_args()


if __name__ == "__main__":
    executar(argumentos())
