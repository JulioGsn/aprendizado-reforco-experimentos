#!/usr/bin/env python3
"""Executa experimentos de saturacao e robustez em ambientes tabulares."""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


ALGORITMOS = ["q_learning", "sarsa"]
CHECKPOINTS_BLACKJACK = [50_000, 100_000, 200_000]
RUIDOS = [0.0, 0.05, 0.10, 0.20]
SEMENTES = list(range(10))
ALPHA = 0.1
GAMMA = 0.99
EPSILON_INICIAL = 1.0
EPSILON_FINAL = 0.05
DECAIMENTO_BLACKJACK = 50_000
AVALIACOES_BLACKJACK = 2_000
AVALIACOES_RUIDO = 500
LIMITE_PASSOS = 1_000


def rotulo_algoritmo(algoritmo):
    return "Q-learning" if algoritmo == "q_learning" else "SARSA"


def epsilon_linear(episodio, horizonte):
    episodio_limitado = min(episodio, horizonte - 1)
    progresso = episodio_limitado / max(horizonte - 1, 1)
    return EPSILON_INICIAL + progresso * (EPSILON_FINAL - EPSILON_INICIAL)


def escolher_acao(q, estado, numero_acoes, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(numero_acoes))
    valores = q.get(estado)
    if valores is None:
        valores = np.zeros(numero_acoes, dtype=np.float64)
    melhores = np.flatnonzero(valores == valores.max())
    return int(rng.choice(melhores))


def atualizar_q(q, algoritmo, estado, acao, recompensa, proximo_estado,
                terminou, proxima_acao):
    if terminou:
        futuro = 0.0
    elif algoritmo == "q_learning":
        futuro = float(q[proximo_estado].max())
    else:
        futuro = float(q[proximo_estado][proxima_acao])
    alvo = float(recompensa) + GAMMA * futuro
    q[estado][acao] += ALPHA * (alvo - q[estado][acao])


def avaliar_blackjack(q, semente, episodios):
    env = gym.make("Blackjack-v1")
    rng = np.random.default_rng(semente + 1_000_000)
    recompensas = []
    passos = []

    for episodio in range(episodios):
        estado, _ = env.reset(seed=semente + 2_000_000 + episodio)
        recompensa_total = 0.0
        total_passos = 0

        while True:
            acao = escolher_acao(q, estado, 2, 0.0, rng)
            estado, recompensa, terminated, truncated, _ = env.step(acao)
            recompensa_total += float(recompensa)
            total_passos += 1
            if terminated or truncated:
                break

        recompensas.append(recompensa_total)
        passos.append(total_passos)

    env.close()
    recompensas = np.asarray(recompensas)
    return {
        "retorno_medio": float(recompensas.mean()),
        "retorno_desvio_episodios": float(recompensas.std(ddof=1)),
        "taxa_vitorias": float(np.mean(recompensas > 0)),
        "taxa_empates": float(np.mean(recompensas == 0)),
        "taxa_derrotas": float(np.mean(recompensas < 0)),
        "passos_medios": float(np.mean(passos)),
    }


def treinar_blackjack(algoritmo, semente):
    env = gym.make("Blackjack-v1")
    q = defaultdict(lambda: np.zeros(2, dtype=np.float64))
    rng = np.random.default_rng(semente)
    avaliacoes = []

    for episodio in range(CHECKPOINTS_BLACKJACK[-1]):
        estado, _ = env.reset(seed=semente + episodio)
        epsilon = epsilon_linear(episodio, DECAIMENTO_BLACKJACK)
        acao = escolher_acao(q, estado, 2, epsilon, rng)

        while True:
            proximo_estado, recompensa, terminated, truncated, _ = env.step(acao)
            terminou = terminated or truncated
            proxima_acao = None
            if not terminou and algoritmo == "sarsa":
                proxima_acao = escolher_acao(
                    q, proximo_estado, 2, epsilon, rng
                )
            atualizar_q(
                q, algoritmo, estado, acao, recompensa, proximo_estado,
                terminou, proxima_acao
            )
            if terminou:
                break
            estado = proximo_estado
            if algoritmo == "q_learning":
                acao = escolher_acao(q, estado, 2, epsilon, rng)
            else:
                acao = proxima_acao

        total_treinado = episodio + 1
        if total_treinado in CHECKPOINTS_BLACKJACK:
            resultado = avaliar_blackjack(q, semente, AVALIACOES_BLACKJACK)
            resultado.update({
                "algoritmo": algoritmo,
                "semente": semente,
                "episodios_treinamento": total_treinado,
                "episodios_avaliacao": AVALIACOES_BLACKJACK,
            })
            avaliacoes.append(resultado)

    env.close()
    return q, avaliacoes


def registrar_politica_blackjack(q, algoritmo, semente):
    registros = []
    for as_utilizavel in [False, True]:
        for soma_jogador in range(12, 22):
            for carta_dealer in range(1, 11):
                estado = (soma_jogador, carta_dealer, as_utilizavel)
                valores = q.get(estado, np.zeros(2, dtype=np.float64))
                registros.append({
                    "algoritmo": algoritmo,
                    "semente": semente,
                    "soma_jogador": soma_jogador,
                    "carta_dealer": carta_dealer,
                    "as_utilizavel": as_utilizavel,
                    "acao": int(np.argmax(valores)),
                })
    return registros


def resumir_blackjack(avaliacoes):
    return (
        avaliacoes.groupby(["algoritmo", "episodios_treinamento"])
        .agg(
            retorno_medio=("retorno_medio", "mean"),
            retorno_desvio_entre_sementes=("retorno_medio", "std"),
            taxa_vitorias_media=("taxa_vitorias", "mean"),
            taxa_empates_media=("taxa_empates", "mean"),
            taxa_derrotas_media=("taxa_derrotas", "mean"),
            passos_medios=("passos_medios", "mean"),
        )
        .reset_index()
    )


def grafico_saturacao_blackjack(resumo, diretorio):
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.8))
    for algoritmo, grupo in resumo.groupby("algoritmo"):
        grupo = grupo.sort_values("episodios_treinamento")
        x = grupo["episodios_treinamento"]
        rotulo = rotulo_algoritmo(algoritmo)
        eixos[0].errorbar(
            x, grupo["retorno_medio"],
            yerr=grupo["retorno_desvio_entre_sementes"], marker="o",
            capsize=4, label=rotulo
        )
        eixos[1].plot(
            x, grupo["taxa_vitorias_media"] * 100,
            marker="o", label=rotulo
        )

    eixos[0].set_title("Retorno medio na avaliacao")
    eixos[0].set_ylabel("Retorno medio")
    eixos[1].set_title("Vitorias na avaliacao")
    eixos[1].set_ylabel("Vitorias (%)")
    for eixo in eixos:
        eixo.set_xlabel("Episodios de treinamento")
        eixo.grid(alpha=0.25)
        eixo.legend()
    fig.suptitle("Blackjack: efeito de continuar o treinamento")
    fig.tight_layout()
    fig.savefig(diretorio / "blackjack_saturacao.png", dpi=180)
    plt.close(fig)


def grafico_politicas_blackjack(politicas, diretorio):
    fig, eixos = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    cmap = ListedColormap(["#2d6a4f", "#d1495b"])

    for linha, algoritmo in enumerate(ALGORITMOS):
        for coluna, as_utilizavel in enumerate([False, True]):
            eixo = eixos[linha, coluna]
            grupo = politicas[
                (politicas["algoritmo"] == algoritmo)
                & (politicas["as_utilizavel"] == as_utilizavel)
            ]
            tabela = (
                grupo.groupby(["soma_jogador", "carta_dealer"])["acao"]
                .mean().unstack()
            )
            modal = (tabela.to_numpy() >= 0.5).astype(int)
            eixo.imshow(modal, cmap=cmap, vmin=0, vmax=1, origin="lower")
            for y in range(modal.shape[0]):
                for x in range(modal.shape[1]):
                    eixo.text(
                        x, y, "P" if modal[y, x] else "F",
                        ha="center", va="center", color="white", fontsize=8
                    )
            eixo.set_xticks(range(10), range(1, 11))
            eixo.set_yticks(range(10), range(12, 22))
            titulo_as = "com as utilizavel" if as_utilizavel else "sem as utilizavel"
            eixo.set_title(f"{rotulo_algoritmo(algoritmo)} - {titulo_as}")
            eixo.set_xlabel("Carta visivel do dealer")
            eixo.set_ylabel("Soma do jogador")

    fig.suptitle("Politica modal em 10 sementes (F=ficar, P=pedir)")
    fig.tight_layout()
    fig.savefig(diretorio / "blackjack_politicas.png", dpi=180)
    plt.close(fig)


def treinar_ambiente(ambiente_id, algoritmo, semente, episodios):
    env = gym.make(ambiente_id)
    numero_acoes = env.action_space.n
    q = defaultdict(lambda: np.zeros(numero_acoes, dtype=np.float64))
    rng = np.random.default_rng(semente)

    for episodio in range(episodios):
        estado, _ = env.reset(seed=semente + episodio)
        epsilon = epsilon_linear(episodio, episodios)
        acao = escolher_acao(q, estado, numero_acoes, epsilon, rng)

        for _ in range(LIMITE_PASSOS):
            proximo_estado, recompensa, terminated, truncated, _ = env.step(acao)
            terminou = terminated or truncated
            proxima_acao = None
            if not terminou and algoritmo == "sarsa":
                proxima_acao = escolher_acao(
                    q, proximo_estado, numero_acoes, epsilon, rng
                )
            atualizar_q(
                q, algoritmo, estado, acao, recompensa, proximo_estado,
                terminou, proxima_acao
            )
            if terminou:
                break
            estado = proximo_estado
            if algoritmo == "q_learning":
                acao = escolher_acao(
                    q, estado, numero_acoes, epsilon, rng
                )
            else:
                acao = proxima_acao

    env.close()
    return q


def avaliar_com_ruido(ambiente_id, q, semente, ruido, episodios):
    env = gym.make(ambiente_id)
    numero_acoes = env.action_space.n
    indice_ruido = RUIDOS.index(ruido)
    rng = np.random.default_rng(semente + 3_000_000 + indice_ruido * 100_000)
    retornos = []
    passos = []
    passos_concluidos = []
    conclusoes = []
    quedas = []
    acoes_ilegais = []
    perturbacoes = []

    for episodio in range(episodios):
        estado, _ = env.reset(seed=semente + 2_000_000 + episodio)
        retorno = 0.0
        total_passos = 0
        total_quedas = 0
        total_ilegais = 0
        total_perturbacoes = 0
        concluiu = False

        for _ in range(LIMITE_PASSOS):
            acao = escolher_acao(q, estado, numero_acoes, 0.0, rng)
            if ruido > 0 and rng.random() < ruido:
                alternativas = np.delete(np.arange(numero_acoes), acao)
                acao = int(rng.choice(alternativas))
                total_perturbacoes += 1
            proximo_estado, recompensa, terminated, truncated, _ = env.step(acao)
            retorno += float(recompensa)
            total_passos += 1
            total_quedas += int(ambiente_id == "CliffWalking-v1" and recompensa == -100)
            total_ilegais += int(ambiente_id == "Taxi-v4" and recompensa == -10)
            estado = proximo_estado
            concluiu = terminated
            if terminated or truncated:
                break

        retornos.append(retorno)
        passos.append(total_passos)
        conclusoes.append(int(concluiu))
        quedas.append(total_quedas)
        acoes_ilegais.append(total_ilegais)
        perturbacoes.append(total_perturbacoes)
        if concluiu:
            passos_concluidos.append(total_passos)

    env.close()
    return {
        "retorno_medio": float(np.mean(retornos)),
        "retorno_desvio_episodios": float(np.std(retornos, ddof=1)),
        "taxa_conclusao": float(np.mean(conclusoes)),
        "passos_medios": float(np.mean(passos)),
        "passos_medios_conclusao": (
            float(np.mean(passos_concluidos)) if passos_concluidos else float("nan")
        ),
        "quedas_medias": float(np.mean(quedas)),
        "acoes_ilegais_medias": float(np.mean(acoes_ilegais)),
        "perturbacoes_medias": float(np.mean(perturbacoes)),
    }


def resumir_ruido(avaliacoes):
    return (
        avaliacoes.groupby(["ambiente", "algoritmo", "ruido"])
        .agg(
            retorno_medio=("retorno_medio", "mean"),
            retorno_desvio_entre_sementes=("retorno_medio", "std"),
            taxa_conclusao_media=("taxa_conclusao", "mean"),
            passos_medios=("passos_medios", "mean"),
            passos_medios_conclusao=("passos_medios_conclusao", "mean"),
            quedas_medias=("quedas_medias", "mean"),
            acoes_ilegais_medias=("acoes_ilegais_medias", "mean"),
            perturbacoes_medias=("perturbacoes_medias", "mean"),
        )
        .reset_index()
    )


def grafico_robustez(resumo, ambiente_id, diretorio):
    metrica_evento = (
        "quedas_medias" if ambiente_id == "CliffWalking-v1"
        else "acoes_ilegais_medias"
    )
    rotulo_evento = (
        "Quedas por episodio" if ambiente_id == "CliffWalking-v1"
        else "Acoes ilegais por episodio"
    )
    fig, eixos = plt.subplots(1, 3, figsize=(15, 4.8))

    for algoritmo, grupo in resumo.groupby("algoritmo"):
        grupo = grupo.sort_values("ruido")
        x = grupo["ruido"] * 100
        rotulo = rotulo_algoritmo(algoritmo)
        eixos[0].errorbar(
            x, grupo["retorno_medio"],
            yerr=grupo["retorno_desvio_entre_sementes"], marker="o",
            capsize=4, label=rotulo
        )
        eixos[1].plot(
            x, grupo["taxa_conclusao_media"] * 100,
            marker="o", label=rotulo
        )
        eixos[2].plot(x, grupo[metrica_evento], marker="o", label=rotulo)

    titulos = ["Retorno medio", "Conclusoes", rotulo_evento]
    rotulos_y = ["Retorno", "Conclusoes (%)", rotulo_evento]
    for eixo, titulo, rotulo_y in zip(eixos, titulos, rotulos_y):
        eixo.set_title(titulo)
        eixo.set_xlabel("Acoes substituidas (%)")
        eixo.set_ylabel(rotulo_y)
        eixo.grid(alpha=0.25)
        eixo.legend()
    fig.suptitle(f"Robustez a ruido nas acoes - {ambiente_id}")
    fig.tight_layout()
    nome = ambiente_id.lower().replace("-", "_")
    fig.savefig(diretorio / f"robustez_{nome}.png", dpi=180)
    plt.close(fig)


def registrar_politica_cliff(q, algoritmo, semente):
    registros = []
    for estado in range(48):
        valores = q.get(estado, np.zeros(4, dtype=np.float64))
        registros.append({
            "algoritmo": algoritmo,
            "semente": semente,
            "estado": estado,
            "linha": estado // 12,
            "coluna": estado % 12,
            "acao": int(np.argmax(valores)),
        })
    return registros


def grafico_politicas_cliff(politicas, diretorio):
    setas = np.array(["↑", "→", "↓", "←"])
    fig, eixos = plt.subplots(2, 1, figsize=(13, 5.5))

    for eixo, algoritmo in zip(eixos, ALGORITMOS):
        grupo = politicas[politicas["algoritmo"] == algoritmo]
        for linha in range(4):
            for coluna in range(12):
                estado = linha * 12 + coluna
                if estado == 36:
                    texto = "S"
                    cor = "#40916c"
                elif estado == 47:
                    texto = "G"
                    cor = "#1d3557"
                elif linha == 3 and 1 <= coluna <= 10:
                    texto = "C"
                    cor = "#d1495b"
                else:
                    acoes = grupo.loc[grupo["estado"] == estado, "acao"]
                    texto = setas[int(acoes.mode().iloc[0])]
                    cor = "#e9ecef"
                retangulo = plt.Rectangle(
                    (coluna, 3 - linha), 1, 1,
                    facecolor=cor, edgecolor="white"
                )
                eixo.add_patch(retangulo)
                eixo.text(
                    coluna + 0.5, 3 - linha + 0.5, texto,
                    ha="center", va="center", fontsize=14
                )
        eixo.set_xlim(0, 12)
        eixo.set_ylim(0, 4)
        eixo.set_aspect("equal")
        eixo.axis("off")
        eixo.set_title(rotulo_algoritmo(algoritmo))

    fig.suptitle("Politica modal no CliffWalking (S=inicio, C=penhasco, G=objetivo)")
    fig.tight_layout()
    fig.savefig(diretorio / "cliffwalking_politicas.png", dpi=180)
    plt.close(fig)


def executar_blackjack(diretorio):
    diretorio.mkdir(parents=True, exist_ok=True)
    avaliacoes = []
    politicas = []
    inicio = time.perf_counter()

    for algoritmo in ALGORITMOS:
        for semente in SEMENTES:
            print(
                f"Blackjack: {algoritmo}, semente {semente}, "
                f"{CHECKPOINTS_BLACKJACK[-1]} episodios...",
                flush=True,
            )
            q, resultados = treinar_blackjack(algoritmo, semente)
            avaliacoes.extend(resultados)
            politicas.extend(registrar_politica_blackjack(q, algoritmo, semente))

    df_avaliacoes = pd.DataFrame(avaliacoes)
    df_politicas = pd.DataFrame(politicas)
    resumo = resumir_blackjack(df_avaliacoes)
    df_avaliacoes.to_csv(diretorio / "avaliacao_por_semente.csv", index=False)
    df_politicas.to_csv(diretorio / "politicas_por_semente.csv", index=False)
    resumo.to_csv(diretorio / "resumo.csv", index=False)
    grafico_saturacao_blackjack(resumo, diretorio)
    grafico_politicas_blackjack(df_politicas, diretorio)
    duracao = time.perf_counter() - inicio

    configuracao = {
        "experimento": "saturacao_blackjack",
        "ambiente": "Blackjack-v1",
        "algoritmos": ALGORITMOS,
        "sementes": SEMENTES,
        "checkpoints_treinamento": CHECKPOINTS_BLACKJACK,
        "episodios_avaliacao": AVALIACOES_BLACKJACK,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "epsilon_inicial": EPSILON_INICIAL,
        "epsilon_final": EPSILON_FINAL,
        "episodios_decaimento_epsilon": DECAIMENTO_BLACKJACK,
        "hipotese_previa": (
            "O desempenho deve se estabilizar porque parte do resultado depende "
            "da aleatoriedade das cartas."
        ),
        "duracao_segundos": duracao,
    }
    salvar_configuracao(diretorio, configuracao)
    print("\nBlackjack:\n", resumo.to_string(index=False), flush=True)


def executar_robustez(ambiente_id, episodios_treinamento, diretorio):
    diretorio.mkdir(parents=True, exist_ok=True)
    avaliacoes = []
    politicas = []
    inicio = time.perf_counter()

    for algoritmo in ALGORITMOS:
        for semente in SEMENTES:
            print(
                f"{ambiente_id}: {algoritmo}, semente {semente}, "
                f"{episodios_treinamento} episodios...",
                flush=True,
            )
            q = treinar_ambiente(
                ambiente_id, algoritmo, semente, episodios_treinamento
            )
            if ambiente_id == "CliffWalking-v1":
                politicas.extend(registrar_politica_cliff(q, algoritmo, semente))
            for ruido in RUIDOS:
                resultado = avaliar_com_ruido(
                    ambiente_id, q, semente, ruido, AVALIACOES_RUIDO
                )
                resultado.update({
                    "ambiente": ambiente_id,
                    "algoritmo": algoritmo,
                    "semente": semente,
                    "ruido": ruido,
                    "episodios_treinamento": episodios_treinamento,
                    "episodios_avaliacao": AVALIACOES_RUIDO,
                })
                avaliacoes.append(resultado)

    df_avaliacoes = pd.DataFrame(avaliacoes)
    resumo = resumir_ruido(df_avaliacoes)
    df_avaliacoes.to_csv(diretorio / "avaliacao_por_semente.csv", index=False)
    resumo.to_csv(diretorio / "resumo.csv", index=False)
    grafico_robustez(resumo, ambiente_id, diretorio)
    if politicas:
        df_politicas = pd.DataFrame(politicas)
        df_politicas.to_csv(diretorio / "politicas_por_semente.csv", index=False)
        grafico_politicas_cliff(df_politicas, diretorio)
    duracao = time.perf_counter() - inicio

    configuracao = {
        "experimento": "robustez_ruido_acoes",
        "ambiente": ambiente_id,
        "algoritmos": ALGORITMOS,
        "sementes": SEMENTES,
        "episodios_treinamento": episodios_treinamento,
        "episodios_avaliacao_por_nivel": AVALIACOES_RUIDO,
        "ruidos": RUIDOS,
        "definicao_ruido": "A acao escolhida e substituida por uma acao diferente.",
        "alpha": ALPHA,
        "gamma": GAMMA,
        "epsilon_inicial": EPSILON_INICIAL,
        "epsilon_final": EPSILON_FINAL,
        "hipotese_previa": (
            "Politicas com desempenho semelhante sem ruido podem responder de "
            "forma diferente a erros ocasionais de execucao."
        ),
        "duracao_segundos": duracao,
    }
    salvar_configuracao(diretorio, configuracao)
    print(f"\n{ambiente_id}:\n", resumo.to_string(index=False), flush=True)


def salvar_configuracao(diretorio, configuracao):
    configuracao.update({
        "python": platform.python_version(),
        "gymnasium": gym.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    })
    (diretorio / "configuracao.json").write_text(
        json.dumps(configuracao, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def argumentos():
    parser = argparse.ArgumentParser(
        description="Experimentos tabulares adicionais para o relatorio."
    )
    parser.add_argument(
        "--saida", default="resultados/tabulares_adicionais",
        help="Diretorio raiz dos resultados.",
    )
    parser.add_argument(
        "--experimentos", nargs="+",
        choices=["blackjack", "cliffwalking", "taxi"],
        default=["blackjack", "cliffwalking", "taxi"],
    )
    return parser.parse_args()


def main():
    args = argumentos()
    raiz = Path(args.saida)
    raiz.mkdir(parents=True, exist_ok=True)
    inicio = time.perf_counter()

    if "blackjack" in args.experimentos:
        executar_blackjack(raiz / "blackjack_saturacao")
    if "cliffwalking" in args.experimentos:
        executar_robustez(
            "CliffWalking-v1", 5_000, raiz / "cliffwalking_robustez"
        )
    if "taxi" in args.experimentos:
        executar_robustez("Taxi-v4", 10_000, raiz / "taxi_robustez")

    print(
        f"\nTempo total: {time.perf_counter() - inicio:.1f} s\n"
        f"Arquivos gravados em: {raiz.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
