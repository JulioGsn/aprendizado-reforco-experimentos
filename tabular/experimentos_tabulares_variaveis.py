#!/usr/bin/env python3
"""Experimentos didaticos sobre gamma e alpha em ambientes tabulares."""

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
import numpy as np
import pandas as pd


ALGORITMOS = ["q_learning", "sarsa"]
GAMMAS = [0.0, 0.5, 0.9, 0.99]
ALPHAS = ["constante_0.1", "decrescente_1/N"]
ALPHA_CONSTANTE = 0.1
GAMMA_BLACKJACK = 0.99
EPSILON_INICIAL = 1.0
EPSILON_FINAL = 0.05
LIMITE_PASSOS = 1_000

PADROES = {
    "teste": {
        "frozenlake_gamma": {"episodios": 300, "sementes": 2, "avaliacoes": 100},
        "blackjack_alpha": {"episodios": 500, "sementes": 2, "avaliacoes": 100},
    },
    "relatorio": {
        "frozenlake_gamma": {
            "episodios": 10_000,
            "sementes": 10,
            "avaliacoes": 1_000,
        },
        "blackjack_alpha": {
            "episodios": 200_000,
            "sementes": 10,
            "avaliacoes": 2_000,
        },
    },
}


def estado_hashable(observacao):
    """Converte observacoes compostas em chaves estaveis da tabela Q."""
    if isinstance(observacao, np.ndarray):
        return tuple(estado_hashable(item) for item in observacao.tolist())
    if isinstance(observacao, (list, tuple)):
        return tuple(estado_hashable(item) for item in observacao)
    if isinstance(observacao, dict):
        return tuple(
            sorted((chave, estado_hashable(valor)) for chave, valor in observacao.items())
        )
    if isinstance(observacao, np.generic):
        return observacao.item()
    return observacao


def valores_q(q, estado, numero_acoes):
    valores = q.get(estado)
    if valores is None:
        return np.zeros(numero_acoes, dtype=np.float64)
    return valores


def escolher_acao(q, estado, numero_acoes, epsilon, rng):
    """Politica epsilon-greedy com desempate aleatorio inclusive na avaliacao."""
    if rng.random() < epsilon:
        return int(rng.integers(numero_acoes))
    valores = valores_q(q, estado, numero_acoes)
    melhores = np.flatnonzero(valores == valores.max())
    return int(rng.choice(melhores))


def epsilon_linear(episodio, total_episodios):
    progresso = episodio / max(total_episodios - 1, 1)
    return EPSILON_INICIAL + progresso * (EPSILON_FINAL - EPSILON_INICIAL)


def atualizar_q(
    q,
    algoritmo,
    estado,
    acao,
    recompensa,
    proximo_estado,
    terminou,
    proxima_acao,
    gamma,
    alpha,
):
    if terminou:
        futuro = 0.0
    elif algoritmo == "q_learning":
        futuro = float(q[proximo_estado].max())
    else:
        futuro = float(q[proximo_estado][proxima_acao])
    alvo = float(recompensa) + gamma * futuro
    q[estado][acao] += alpha * (alvo - q[estado][acao])


def avaliar_frozenlake(q, semente, episodios):
    env = gym.make("FrozenLake-v1")
    numero_acoes = env.action_space.n
    rng = np.random.default_rng(semente + 1_000_000)
    retornos = []
    sucessos = []
    passos = []

    for episodio in range(episodios):
        estado, _ = env.reset(seed=semente + 2_000_000 + episodio)
        estado = estado_hashable(estado)
        retorno = 0.0

        for passo in range(1, LIMITE_PASSOS + 1):
            acao = escolher_acao(q, estado, numero_acoes, 0.0, rng)
            observacao, recompensa, terminated, truncated, _ = env.step(acao)
            estado = estado_hashable(observacao)
            retorno += float(recompensa)
            if terminated or truncated:
                break

        retornos.append(retorno)
        sucessos.append(int(retorno == 1.0))
        passos.append(passo)

    env.close()
    return {
        "retorno_medio": float(np.mean(retornos)),
        "retorno_desvio_episodios": float(np.std(retornos, ddof=1)),
        "taxa_sucesso": float(np.mean(sucessos)),
        "passos_medios": float(np.mean(passos)),
    }


def treinar_frozenlake(algoritmo, gamma, semente, episodios, avaliacoes):
    env = gym.make("FrozenLake-v1")
    numero_acoes = env.action_space.n
    q = defaultdict(lambda: np.zeros(numero_acoes, dtype=np.float64))
    rng = np.random.default_rng(semente)
    registros = []
    inicio = time.perf_counter()

    for episodio in range(episodios):
        estado, _ = env.reset(seed=semente + episodio)
        estado = estado_hashable(estado)
        epsilon = epsilon_linear(episodio, episodios)
        acao = escolher_acao(q, estado, numero_acoes, epsilon, rng)
        retorno = 0.0

        for passo in range(1, LIMITE_PASSOS + 1):
            observacao, recompensa, terminated, truncated, _ = env.step(acao)
            proximo_estado = estado_hashable(observacao)
            terminou = terminated or truncated
            proxima_acao = None
            if not terminou and algoritmo == "sarsa":
                proxima_acao = escolher_acao(
                    q, proximo_estado, numero_acoes, epsilon, rng
                )
            atualizar_q(
                q,
                algoritmo,
                estado,
                acao,
                recompensa,
                proximo_estado,
                terminou,
                proxima_acao,
                gamma,
                ALPHA_CONSTANTE,
            )
            retorno += float(recompensa)
            if terminou:
                break
            estado = proximo_estado
            acao = (
                escolher_acao(q, estado, numero_acoes, epsilon, rng)
                if algoritmo == "q_learning"
                else proxima_acao
            )

        registros.append(
            {
                "algoritmo": algoritmo,
                "gamma": gamma,
                "semente": semente,
                "episodio": episodio + 1,
                "epsilon": epsilon,
                "retorno": retorno,
                "sucesso": int(retorno == 1.0),
                "passos": passo,
            }
        )

    duracao = time.perf_counter() - inicio
    env.close()
    resultado = avaliar_frozenlake(q, semente, avaliacoes)
    resultado.update(
        {
            "algoritmo": algoritmo,
            "gamma": gamma,
            "semente": semente,
            "episodios_treinamento": episodios,
            "episodios_avaliacao": avaliacoes,
            "duracao_treinamento_segundos": duracao,
            "estados_na_tabela_q": len(q),
        }
    )
    return registros, resultado


def avaliar_blackjack(q, semente, episodios, checkpoint):
    env = gym.make("Blackjack-v1")
    numero_acoes = env.action_space.n
    rng = np.random.default_rng(semente + 1_000_000 + checkpoint)
    retornos = []
    passos = []

    for episodio in range(episodios):
        estado, _ = env.reset(seed=semente + 2_000_000 + episodio)
        estado = estado_hashable(estado)
        retorno = 0.0

        for passo in range(1, LIMITE_PASSOS + 1):
            acao = escolher_acao(q, estado, numero_acoes, 0.0, rng)
            observacao, recompensa, terminated, truncated, _ = env.step(acao)
            estado = estado_hashable(observacao)
            retorno += float(recompensa)
            if terminated or truncated:
                break

        retornos.append(retorno)
        passos.append(passo)

    env.close()
    retornos = np.asarray(retornos)
    return {
        "retorno_medio": float(retornos.mean()),
        "retorno_desvio_episodios": float(retornos.std(ddof=1)),
        "taxa_vitorias": float(np.mean(retornos > 0)),
        "taxa_empates": float(np.mean(retornos == 0)),
        "taxa_derrotas": float(np.mean(retornos < 0)),
        "passos_medios": float(np.mean(passos)),
    }


def checkpoints_para(total_episodios):
    return sorted(
        {
            max(1, total_episodios // 4),
            max(1, total_episodios // 2),
            total_episodios,
        }
    )


def treinar_blackjack(
    algoritmo, esquema_alpha, semente, episodios, checkpoints, avaliacoes
):
    env = gym.make("Blackjack-v1")
    numero_acoes = env.action_space.n
    q = defaultdict(lambda: np.zeros(numero_acoes, dtype=np.float64))
    visitas = defaultdict(int)
    rng = np.random.default_rng(semente)
    registros_checkpoints = []
    resultados_avaliacao = []
    retornos_intervalo = []
    soma_alphas = 0.0
    total_atualizacoes = 0
    inicio = time.perf_counter()

    for episodio in range(episodios):
        estado, _ = env.reset(seed=semente + episodio)
        estado = estado_hashable(estado)
        epsilon = epsilon_linear(episodio, episodios)
        acao = escolher_acao(q, estado, numero_acoes, epsilon, rng)
        retorno = 0.0

        for _ in range(LIMITE_PASSOS):
            observacao, recompensa, terminated, truncated, _ = env.step(acao)
            proximo_estado = estado_hashable(observacao)
            terminou = terminated or truncated
            proxima_acao = None
            if not terminou and algoritmo == "sarsa":
                proxima_acao = escolher_acao(
                    q, proximo_estado, numero_acoes, epsilon, rng
                )

            par = (estado, acao)
            visitas[par] += 1
            alpha = (
                ALPHA_CONSTANTE
                if esquema_alpha == "constante_0.1"
                else 1.0 / visitas[par]
            )
            atualizar_q(
                q,
                algoritmo,
                estado,
                acao,
                recompensa,
                proximo_estado,
                terminou,
                proxima_acao,
                GAMMA_BLACKJACK,
                alpha,
            )
            retorno += float(recompensa)
            soma_alphas += alpha
            total_atualizacoes += 1
            if terminou:
                break
            estado = proximo_estado
            acao = (
                escolher_acao(q, estado, numero_acoes, epsilon, rng)
                if algoritmo == "q_learning"
                else proxima_acao
            )

        retornos_intervalo.append(retorno)
        total_treinado = episodio + 1
        if total_treinado in checkpoints:
            contagens = np.asarray(list(visitas.values()), dtype=np.float64)
            registros_checkpoints.append(
                {
                    "algoritmo": algoritmo,
                    "esquema_alpha": esquema_alpha,
                    "semente": semente,
                    "checkpoint_episodios": total_treinado,
                    "epsilon": epsilon,
                    "retorno_medio_treino_desde_checkpoint": float(
                        np.mean(retornos_intervalo)
                    ),
                    "atualizacoes_acumuladas": total_atualizacoes,
                    "alpha_efetivo_medio_acumulado": soma_alphas
                    / total_atualizacoes,
                    "pares_estado_acao_visitados": len(visitas),
                    "visitas_medias_por_par": float(contagens.mean()),
                    "visitas_maximas_em_um_par": int(contagens.max()),
                }
            )
            retornos_intervalo = []
            resultado = avaliar_blackjack(
                q, semente, avaliacoes, total_treinado
            )
            resultado.update(
                {
                    "algoritmo": algoritmo,
                    "esquema_alpha": esquema_alpha,
                    "semente": semente,
                    "checkpoint_episodios": total_treinado,
                    "episodios_avaliacao": avaliacoes,
                    "estados_na_tabela_q": len(q),
                }
            )
            resultados_avaliacao.append(resultado)

    duracao = time.perf_counter() - inicio
    env.close()
    for resultado in resultados_avaliacao:
        resultado["duracao_treinamento_total_segundos"] = duracao
    return registros_checkpoints, resultados_avaliacao


def resumir_frozenlake(avaliacoes):
    return (
        avaliacoes.groupby(["algoritmo", "gamma"])
        .agg(
            retorno_medio=("retorno_medio", "mean"),
            retorno_desvio_entre_sementes=("retorno_medio", "std"),
            taxa_sucesso_media=("taxa_sucesso", "mean"),
            taxa_sucesso_desvio_entre_sementes=("taxa_sucesso", "std"),
            passos_medios=("passos_medios", "mean"),
            duracao_media_segundos=("duracao_treinamento_segundos", "mean"),
        )
        .reset_index()
    )


def resumir_blackjack(avaliacoes):
    return (
        avaliacoes.groupby(
            ["algoritmo", "esquema_alpha", "checkpoint_episodios"]
        )
        .agg(
            retorno_medio=("retorno_medio", "mean"),
            retorno_desvio_entre_sementes=("retorno_medio", "std"),
            taxa_vitorias_media=("taxa_vitorias", "mean"),
            taxa_vitorias_desvio_entre_sementes=("taxa_vitorias", "std"),
            taxa_empates_media=("taxa_empates", "mean"),
            taxa_empates_desvio_entre_sementes=("taxa_empates", "std"),
            taxa_derrotas_media=("taxa_derrotas", "mean"),
            taxa_derrotas_desvio_entre_sementes=("taxa_derrotas", "std"),
        )
        .reset_index()
    )


def grafico_frozenlake_resumo(resumo, diretorio):
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.8))
    for algoritmo, grupo in resumo.groupby("algoritmo"):
        grupo = grupo.sort_values("gamma")
        rotulo = "Q-learning" if algoritmo == "q_learning" else "SARSA"
        eixos[0].errorbar(
            grupo["gamma"],
            grupo["taxa_sucesso_media"] * 100,
            yerr=grupo["taxa_sucesso_desvio_entre_sementes"] * 100,
            marker="o",
            capsize=4,
            label=rotulo,
        )
        eixos[1].errorbar(
            grupo["gamma"],
            grupo["retorno_medio"],
            yerr=grupo["retorno_desvio_entre_sementes"],
            marker="o",
            capsize=4,
            label=rotulo,
        )
    eixos[0].set_title("Sucesso na avaliacao")
    eixos[0].set_ylabel("Episodios com recompensa 1 (%)")
    eixos[1].set_title("Retorno na avaliacao")
    eixos[1].set_ylabel("Retorno medio")
    for eixo in eixos:
        eixo.set_xlabel("Gamma")
        eixo.set_xticks(GAMMAS)
        eixo.grid(alpha=0.25)
        eixo.legend()
    fig.suptitle("FrozenLake: quanto o futuro importa?")
    fig.tight_layout()
    fig.savefig(diretorio / "comparacao_gamma_avaliacao.png", dpi=180)
    plt.close(fig)


def grafico_frozenlake_treino(episodios, diretorio):
    plt.figure(figsize=(11, 6))
    janela = max(10, int(episodios["episodio"].max()) // 50)
    agrupado = episodios.groupby(["algoritmo", "gamma", "episodio"])["sucesso"].mean()
    for (algoritmo, gamma), serie in agrupado.groupby(level=[0, 1]):
        serie = serie.droplevel([0, 1]).rolling(janela, min_periods=1).mean()
        rotulo_algoritmo = "Q" if algoritmo == "q_learning" else "SARSA"
        plt.plot(serie.index, serie * 100, label=f"{rotulo_algoritmo}, gamma={gamma}")
    plt.title("FrozenLake: sucesso durante o treinamento")
    plt.xlabel("Episodio")
    plt.ylabel(f"Sucesso medio movel (%) - janela {janela}")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(diretorio / "curvas_treinamento_gamma.png", dpi=180)
    plt.close()


def grafico_blackjack(resumo, diretorio):
    fig, eixos = plt.subplots(1, 2, figsize=(13, 5))
    for (algoritmo, esquema), grupo in resumo.groupby(
        ["algoritmo", "esquema_alpha"]
    ):
        grupo = grupo.sort_values("checkpoint_episodios")
        nome_algoritmo = "Q-learning" if algoritmo == "q_learning" else "SARSA"
        nome_alpha = "alpha=0.1" if esquema == "constante_0.1" else "alpha=1/N(s,a)"
        rotulo = f"{nome_algoritmo}, {nome_alpha}"
        eixos[0].errorbar(
            grupo["checkpoint_episodios"],
            grupo["retorno_medio"],
            yerr=grupo["retorno_desvio_entre_sementes"],
            marker="o",
            capsize=4,
            label=rotulo,
        )
        eixos[1].plot(
            grupo["checkpoint_episodios"],
            grupo["taxa_vitorias_media"] * 100,
            marker="o",
            label=rotulo,
        )
    eixos[0].set_title("Retorno sem exploracao")
    eixos[0].set_ylabel("Retorno medio")
    eixos[1].set_title("Vitorias sem exploracao")
    eixos[1].set_ylabel("Vitorias (%)")
    for eixo in eixos:
        eixo.set_xlabel("Episodios treinados (checkpoint)")
        eixo.grid(alpha=0.25)
        eixo.legend(fontsize=8)
    fig.suptitle("Blackjack: alpha constante ou decrescente?")
    fig.tight_layout()
    fig.savefig(diretorio / "comparacao_alpha_checkpoints.png", dpi=180)
    plt.close(fig)


def grafico_blackjack_desfechos(resumo, diretorio):
    checkpoint = int(resumo["checkpoint_episodios"].max())
    final = resumo[resumo["checkpoint_episodios"] == checkpoint].copy()
    final["configuracao"] = final.apply(
        lambda linha: (
            ("Q" if linha["algoritmo"] == "q_learning" else "SARSA")
            + "\n"
            + ("0.1" if linha["esquema_alpha"] == "constante_0.1" else "1/N")
        ),
        axis=1,
    )
    x = np.arange(len(final))
    base = np.zeros(len(final))
    plt.figure(figsize=(9, 5))
    for coluna, rotulo, cor in [
        ("taxa_vitorias_media", "Vitoria", "#2a9d8f"),
        ("taxa_empates_media", "Empate", "#e9c46a"),
        ("taxa_derrotas_media", "Derrota", "#e76f51"),
    ]:
        valores = final[coluna].to_numpy() * 100
        plt.bar(x, valores, bottom=base, label=rotulo, color=cor)
        base += valores
    plt.xticks(x, final["configuracao"])
    plt.ylabel("Episodios de avaliacao (%)")
    plt.title(f"Blackjack: desfechos apos {checkpoint} episodios")
    plt.legend()
    plt.tight_layout()
    plt.savefig(diretorio / "desfechos_blackjack_checkpoint_final.png", dpi=180)
    plt.close()


def metadados_base(args, episodios, sementes, avaliacoes, diretorio):
    return {
        "experimento": args.experimento,
        "pergunta": (
            "Quanto o futuro importa numa recompensa distante?"
            if args.experimento == "frozenlake_gamma"
            else "Alpha decrescente estabiliza um ambiente estocastico?"
        ),
        "modo": args.modo,
        "algoritmos": ALGORITMOS,
        "sementes": list(range(sementes)),
        "episodios_treinamento": episodios,
        "episodios_avaliacao": avaliacoes,
        "epsilon_inicial": EPSILON_INICIAL,
        "epsilon_final": EPSILON_FINAL,
        "agenda_epsilon": "linear ao longo de todo o treinamento",
        "limite_passos": LIMITE_PASSOS,
        "protocolo_seeds": {
            "rng_politica_treino": "semente",
            "reset_treino": "semente + indice_do_episodio",
            "reset_avaliacao": "semente + 2000000 + indice_do_episodio",
            "rng_desempate_avaliacao_frozenlake": "semente + 1000000",
            "rng_desempate_avaliacao_blackjack": (
                "semente + 1000000 + checkpoint"
            ),
        },
        "saida": str(diretorio.resolve()),
        "python": platform.python_version(),
        "gymnasium": gym.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def executar_frozenlake(args, diretorio, episodios, sementes, avaliacoes):
    registros = []
    resultados = []
    for algoritmo in ALGORITMOS:
        for gamma in GAMMAS:
            for semente in range(sementes):
                print(
                    f"FrozenLake: {algoritmo}, gamma={gamma}, semente={semente}, "
                    f"episodios={episodios}",
                    flush=True,
                )
                linhas, resultado = treinar_frozenlake(
                    algoritmo, gamma, semente, episodios, avaliacoes
                )
                registros.extend(linhas)
                resultados.append(resultado)

    df_episodios = pd.DataFrame(registros)
    df_avaliacoes = pd.DataFrame(resultados)
    resumo = resumir_frozenlake(df_avaliacoes)
    df_episodios.to_csv(diretorio / "episodios_treinamento.csv", index=False)
    df_avaliacoes.to_csv(diretorio / "avaliacao_por_semente.csv", index=False)
    resumo.to_csv(diretorio / "resumo_agregado.csv", index=False)
    grafico_frozenlake_treino(df_episodios, diretorio)
    grafico_frozenlake_resumo(resumo, diretorio)
    return resumo, {
        "ambiente": "FrozenLake-v1",
        "gammas": GAMMAS,
        "alpha": ALPHA_CONSTANTE,
        "definicao_sucesso": "retorno do episodio exatamente igual a 1",
    }


def executar_blackjack(args, diretorio, episodios, sementes, avaliacoes):
    checkpoints = checkpoints_para(episodios)
    registros = []
    resultados = []
    for algoritmo in ALGORITMOS:
        for esquema in ALPHAS:
            for semente in range(sementes):
                print(
                    f"Blackjack: {algoritmo}, {esquema}, semente={semente}, "
                    f"episodios={episodios}",
                    flush=True,
                )
                linhas, avaliacoes_semente = treinar_blackjack(
                    algoritmo,
                    esquema,
                    semente,
                    episodios,
                    checkpoints,
                    avaliacoes,
                )
                registros.extend(linhas)
                resultados.extend(avaliacoes_semente)

    df_checkpoints = pd.DataFrame(registros)
    df_avaliacoes = pd.DataFrame(resultados)
    resumo = resumir_blackjack(df_avaliacoes)
    df_checkpoints.to_csv(diretorio / "checkpoints_treinamento.csv", index=False)
    df_avaliacoes.to_csv(diretorio / "avaliacao_por_semente.csv", index=False)
    resumo.to_csv(diretorio / "resumo_agregado.csv", index=False)
    grafico_blackjack(resumo, diretorio)
    grafico_blackjack_desfechos(resumo, diretorio)
    return resumo, {
        "ambiente": "Blackjack-v1",
        "checkpoints": checkpoints,
        "gamma": GAMMA_BLACKJACK,
        "esquemas_alpha": {
            "constante_0.1": "alpha=0.1 em toda atualizacao",
            "decrescente_1/N": (
                "alpha=1/N(s,a), onde N inclui a visita da atualizacao atual"
            ),
        },
        "metricas_desfecho": ["vitoria", "empate", "derrota", "retorno"],
    }


def inteiro_positivo(valor):
    numero = int(valor)
    if numero <= 0:
        raise argparse.ArgumentTypeError("o valor deve ser um inteiro positivo")
    return numero


def argumentos():
    parser = argparse.ArgumentParser(
        description="Varia gamma no FrozenLake ou alpha no Blackjack."
    )
    parser.add_argument(
        "--experimento",
        required=True,
        choices=["frozenlake_gamma", "blackjack_alpha"],
    )
    parser.add_argument("--modo", choices=["teste", "relatorio"], default="teste")
    parser.add_argument("--episodios", type=inteiro_positivo)
    parser.add_argument("--sementes", type=inteiro_positivo)
    parser.add_argument("--episodios-avaliacao", type=inteiro_positivo)
    parser.add_argument(
        "--saida",
        help="Diretorio de saida; por padrao, resultados_variaveis/<experimento>.",
    )
    return parser.parse_args()


def main():
    args = argumentos()
    padrao = PADROES[args.modo][args.experimento]
    episodios = args.episodios or padrao["episodios"]
    sementes = args.sementes or padrao["sementes"]
    avaliacoes = args.episodios_avaliacao or padrao["avaliacoes"]
    diretorio = Path(args.saida or f"resultados_variaveis/{args.experimento}")
    diretorio.mkdir(parents=True, exist_ok=True)
    inicio = time.perf_counter()

    if args.experimento == "frozenlake_gamma":
        resumo, especifico = executar_frozenlake(
            args, diretorio, episodios, sementes, avaliacoes
        )
    else:
        resumo, especifico = executar_blackjack(
            args, diretorio, episodios, sementes, avaliacoes
        )

    configuracao = metadados_base(
        args, episodios, sementes, avaliacoes, diretorio
    )
    configuracao.update(especifico)
    configuracao["duracao_total_segundos"] = time.perf_counter() - inicio
    (diretorio / "configuracao.json").write_text(
        json.dumps(configuracao, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nResumo agregado:")
    print(resumo.to_string(index=False))
    print(f"\nArquivos gravados em: {diretorio.resolve()}")


if __name__ == "__main__":
    main()
