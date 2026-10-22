# Experimentos de Aprendizado por Reforço

Códigos utilizados no relatório da disciplina Aprendizado por Reforço do PPGTI - IMD/UFRN. O projeto compara Q-learning, SARSA, DQN e Double DQN em ambientes tabulares, de controle clássico e Atari.

## Estrutura

```text
tabular/             Scripts executados localmente
notebooks/           Notebooks executados no Google Colab
resultados_resumo/   CSVs agregados usados nas tabelas do relatório
requirements-tabular.txt
```

## Experimentos tabulares locais

Crie um ambiente virtual e instale as dependências:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-tabular.txt
```

Comparações-base em FrozenLake, Blackjack, CliffWalking e Taxi:

```bash
python tabular/experimentos_tabulares.py --modo relatorio
```

Saturação do Blackjack e robustez a ruído no CliffWalking e Taxi:

```bash
python tabular/experimentos_tabulares_adicionais.py
```

Variação de `gamma` no FrozenLake:

```bash
python tabular/experimentos_tabulares_variaveis.py \
  --experimento frozenlake_gamma --modo relatorio
```

Comparação de `alpha` no Blackjack:

```bash
python tabular/experimentos_tabulares_variaveis.py \
  --experimento blackjack_alpha --modo relatorio
```

Os scripts registram configurações, sementes, resultados por execução, resumos agregados e figuras.

## DQN e DDQN no Google Colab

O notebook `notebooks/dqn_ddqn_classic_control_colab.ipynb` contém os experimentos usados em CartPole e MountainCar. Para reproduzir as configurações do relatório, execute separadamente:

- `baseline`;
- `exploracao_rapida`;
- `target_rapido`;
- `target_lento`;
- `treino_longo`, alterando o orçamento de CartPole para 1.000 episódios.

O notebook utiliza as sementes `0`, `1` e `2` e 20 episódios de avaliação por modelo. Os resultados foram obtidos no Google Colab com GPU.

## DQN e DDQN no Space Invaders

O notebook `notebooks/dqn_ddqn_space_invaders_colab.ipynb` implementa o pré-processamento Atari, CNN, replay buffer, avaliações periódicas e checkpoints. O fluxo é:

1. selecionar GPU no Colab;
2. executar o modo `smoke` para validar ambiente, ROM e shapes;
3. alterar para `MODO = "relatorio"`;
4. executar 200.000 passos para DQN e DDQN;
5. repetir para as sementes `42`, `43`, `44`, `45` e `46`.

Cada passo do agente avança até quatro frames do emulador. As ROMs do Atari são instaladas pelo AutoROM mediante aceitação da licença correspondente.

## Resultados incluídos

`resultados_resumo/` contém somente tabelas agregadas e leves. Os arquivos brutos completos e checkpoints não foram incluídos devido ao tamanho. A semente é a unidade de repetição usada nos resumos entre execuções.

## Reprodutibilidade

- As avaliações usam política sem exploração (`epsilon = 0`).
- Algoritmos comparados recebem as mesmas sementes e o mesmo orçamento.
- As versões locais estão fixadas em `requirements-tabular.txt`.
- Os notebooks instalam as dependências necessárias nas primeiras células.
- Resultados podem apresentar pequenas diferenças entre versões de bibliotecas e hardware.

## Autor

Júlio Gomes da Silva Neto  
PPGTI - IMD/UFRN
