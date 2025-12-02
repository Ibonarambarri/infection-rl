# Infection RL

Entorno multi-agente donde agentes sanos (HealthyAgent) huyen de agentes infectados (InfectedAgent).

## Estructura

```
src/
├── agents/              # Clases de agentes
│   ├── base_agent.py    # Clase base abstracta
│   ├── healthy_agent.py # Agente sano
│   ├── infected_agent.py# Agente infectado
│   └── agent_collection.py
├── envs/
│   ├── environment.py   # Entorno Gymnasium
│   ├── map_generator.py # Carga mapas
│   └── wrappers.py      # Wrappers para SB3

scripts/
├── train.py             # Entrenar (PPO/A2C/DQN)
├── train_dual.py        # Entrenar ambos roles
├── play.py              # Visualizar partidas
└── evaluate.py          # Evaluar modelos

maps/
└── ciudad_60x60.txt     # Mapa predefinido
```

## Instalacion

```bash
pip install stable-baselines3 gymnasium pygame numpy matplotlib
```

## Comandos

### Entrenar

```bash
# Agente sano
python scripts/train.py --role healthy --algo ppo --timesteps 500000

# Agente infectado
python scripts/train.py --role infected --algo ppo --timesteps 500000

# Ambos alternadamente
python scripts/train_dual.py --timesteps 500000
```

### Visualizar

```bash
python scripts/play.py --models-dir models/dual_xxx
```

### Evaluar

```bash
python scripts/evaluate.py --model models/xxx/best_model.zip --episodes 100
```

## Publicar en GitHub

```bash
# 1. Inicializar repositorio
cd /path/to/RL
git init

# 2. Añadir archivos
git add .

# 3. Primer commit
git commit -m "Initial commit: Infection RL environment"

# 4. Crear repositorio en GitHub (desde web o gh cli)
# Opcion A: Desde github.com -> New repository
# Opcion B: gh repo create infection-rl --public

# 5. Conectar y subir
git remote add origin https://github.com/TU_USUARIO/infection-rl.git
git branch -M main
git push -u origin main
```
