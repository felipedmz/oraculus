# Oraculus

Bot for cryptocurrency prediction

## Setup

1. Criação de um ambiente conda e instalação das dependências do projeto

```shell
conda create -n NOME_DO_AMBIENTE --file requirements.txt
```

2. Ativação do ambiente 

```shell
conda activate NOME_DO_AMBIENTE
```

Para mais informações sobre instalação ou gestão de ambientes consulte a documentação oficial do [Anaconda](https://docs.anaconda.com/free/anaconda/install/windows/) ou [conda cheat sheet](https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf)

3. Crawler de dados

```shell
python get_data.py
```

4. Execução de trades

```shell
python main.py
```
