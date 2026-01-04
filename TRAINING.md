# Infection RL - Documentacion de Entrenamiento

## Resumen del Proyecto

Infection RL es un entorno multi-agente donde agentes **healthy** (sanos) intentan sobrevivir mientras agentes **infected** (infectados) intentan cazarlos e infectarlos. Cuando un healthy es alcanzado por un infected, se convierte en infected.

### Mecanicas del Juego

- **Mapa**: Grid 2D con paredes y obstaculos
- **Agentes**:
  - Healthy: Huyen de los infected
  - Infected: Cazan a los healthy (con ventajas de velocidad y vision)
- **Victoria**:
  - Infected ganan si infectan a todos los healthy
  - Healthy ganan si sobreviven hasta max_steps (500)

---

## Ventajas de los Infectados

Para balancear el juego (los infected son numericamente inferiores), tienen dos ventajas:

### Velocidad Doble

Los infectados se mueven **2 celdas por accion** mientras los healthy solo 1.

```
Configuracion: infected_speed = 2

Healthy:  1 celda por "forward"
Infected: 2 celdas por "forward" (si no hay obstaculo)
```

### Vision Global

Los infectados conocen la posicion de **TODOS** los healthy en el mapa, no solo los cercanos.

```
Configuracion: infected_global_vision = True

Healthy:  nearby_agents muestra los 8 agentes mas cercanos
Infected: nearby_agents prioriza TODOS los healthy (hasta 8)
```

Esto simula que los infectados tienen "sentido del olfato" o capacidad de detectar presas.

---

## Sistema de Observacion

### Vista Circular (v3)

Los agentes tienen una vista circular de **radio 7** (15x15 celdas) centrada en su posicion. La vista es independiente de la direccion del agente (vision 360).

```
    +-------------------------------+
    |  .  .  .  .  .  .  .  .  .  . |
    |  .  .  .  .  .  .  .  .  .  . |
    |  .  .  .  .  .  .  .  .  .  . |
    |  .  .  .  .  .  .  .  .  .  . |
    |  .  .  .  .  .  @  .  .  .  . |  <- Agente en el CENTRO
    |  .  .  .  .  .  .  .  .  .  . |
    |  .  .  .  .  .  .  .  .  .  . |
    |  .  .  .  .  .  .  .  .  .  . |
    |  .  .  .  .  .  .  .  .  .  . |
    +-------------------------------+
         Radio = 7, Diametro = 15
```

### Canales de la Vista

La vista tiene **2 canales**:

| Canal | Contenido | Valores |
|-------|-----------|---------|
| 0 | Tipo de celda | 0=vacio, 1=bloqueado, 2=sano, 3=infectado, 4=self |
| 1 | Distancia Manhattan | 0-255 (normalizada) |

**Nota**: Muros y obstaculos estan unificados como "bloqueado" (valor 1) para simplificar la observacion.

### Estructura Completa de Observacion

```
image:          15 x 15 x 2 = 450 valores
direction:      4 (one-hot encoding)
state:          2 (one-hot: 0=healthy, 1=infected)
position:       2 (x, y normalizados)
nearby_agents:  8 x 4 = 32 (8 agentes mas cercanos)
---------------------------------------------
TOTAL:          490 valores
```

---

## Sistema de Rewards: ULTRA-SPARSE

El entrenamiento usa rewards **puramente sparse** - solo se da reward al final del episodio basado en victoria/derrota.

### Filosofia

```
+-------------------------------------------------------------+
|  ULTRA-SPARSE REWARDS                                        |
+-------------------------------------------------------------+
|                                                             |
|  Infected:                                                   |
|    - Ganan (infectan a todos) -> +10.0                      |
|    - Pierden (tiempo agotado) -> 0.0                        |
|                                                             |
|  Healthy:                                                    |
|    - Ganan (sobreviven)       -> +10.0                      |
|    - Pierden (infectados)     -> -10.0                      |
|                                                             |
+-------------------------------------------------------------+
```

### Configuracion de Rewards

| Reward | Healthy | Infected |
|--------|---------|----------|
| survive_step | 0.0 | - |
| distance_bonus | 0.0 | - |
| infected_penalty | **-10.0** | - |
| survive_episode | **+10.0** | - |
| infect_agent | - | 0.0 |
| approach_bonus | - | 0.0 |
| progress_bonus | - | 0.0 |
| no_progress_penalty | - | 0.0 |
| all_infected_bonus | - | **+10.0** |

### Por que Sparse?

Los rewards densos (approach_bonus, progress_bonus, etc.) causaban que los infected aprendieran a "orbitar" cerca de los healthy sin infectarlos, maximizando el approach_bonus.

Con sparse rewards, la UNICA forma de obtener reward es GANAR el episodio.

---

## Estructura del Curriculum (train_sparse.py)

```
FASE 1: MAP_LVL1 (20x20)
+-- 4 healthy vs 1 infected
+-- Rewards: ULTRA-SPARSE
+-- Infected: 500k steps, Healthy: 300k steps
+-- max_steps: 300

FASE 2: MAP_LVL2 (30x30)
+-- 6 healthy vs 2 infected
+-- Rewards: ULTRA-SPARSE
+-- Infected: 700k steps, Healthy: 500k steps
+-- max_steps: 400

FASE 3: MAP_LVL3 (40x40)
+-- 8 healthy vs 2 infected
+-- Rewards: ULTRA-SPARSE
+-- Infected: 1M steps, Healthy: 700k steps
+-- max_steps: 500

REFINAMIENTO ADAPTATIVO: 10 ciclos
+-- 8 healthy vs 2 infected en MAP_LVL3
+-- Rewards: ULTRA-SPARSE
+-- 300k steps por ciclo
+-- Solo entrena al modelo que pierde
```

### Orden de Entrenamiento

En cada fase se entrena **Infected primero**, luego Healthy. Esto permite que los healthy aprendan contra un oponente ya entrenado.

---

## Refinamiento Adaptativo

La fase final usa un sistema adaptativo que solo entrena al modelo mas debil:

```
+---------------------------------------------+
|  CICLO DE REFINAMIENTO ADAPTATIVO           |
+---------------------------------------------+
|  1. Evaluar ambos modelos (20 episodios)    |
|  2. Determinar quien esta perdiendo         |
|  3. Entrenar SOLO al peor (300k steps)      |
|  4. Repetir hasta 10 ciclos                 |
+---------------------------------------------+
```

### Criterio de Decision

| Win Rate | Accion |
|----------|--------|
| Infected > 60% | Entrenar Healthy |
| Healthy > 60% | Entrenar Infected |
| Equilibrado (40-60%) | Entrenar Infected (desventaja numerica) |

---

## Uso

### Entrenar (Sparse - Recomendado)

```bash
# Entrenamiento sparse con curriculum
python scripts/train_sparse.py --output-dir models/sparse_v1

# Con visualizacion de evaluaciones
python scripts/train_sparse.py --output-dir models/sparse_v1 --render

# Desde una fase especifica
python scripts/train_sparse.py --output-dir models/sparse_v1 --start-phase 2
```

### Entrenar (Dense - Legacy)

```bash
# Entrenamiento con rewards densos (no recomendado)
python scripts/train.py --output-dir models/dense_v1
```

### Visualizar Partida

```bash
# Nivel 3 (40x40) con modelos por defecto
python scripts/play.py -l 3

# Nivel 1 con modelos custom
python scripts/play.py -l 1 -m models/sparse_v1

# Con semilla especifica
python scripts/play.py -l 2 --seed 123
```

Controles durante el juego:
- **SPACE**: Pausar/Reanudar
- **V**: Toggle vision de infectados
- **S**: Paso a paso (cuando pausado)
- **Q**: Salir
- **1-5**: Velocidad

---

## Archivos Principales

| Archivo | Descripcion |
|---------|-------------|
| `src/envs/environment.py` | Entorno principal, vista circular, velocidad infected |
| `src/envs/reward_config.py` | Configuracion de rewards por preset |
| `src/envs/wrappers.py` | Wrappers para SB3, soporte infected_speed/global_vision |
| `scripts/train_sparse.py` | **Script de entrenamiento sparse (recomendado)** |
| `scripts/train.py` | Script de entrenamiento dense (legacy) |
| `scripts/play.py` | Visualizador de partidas |

---

## Notas Tecnicas

### Parameter Sharing

El entrenamiento usa Parameter Sharing: todos los agentes del mismo rol comparten la misma red neuronal. Con 8 healthy vs 2 infected, los healthy tienen 4x mas experiencias por timestep.

Compensamos dando **~1.5x mas timesteps** a los infected.

### PPO para Sparse Rewards

Parametros optimizados para sparse rewards:

```python
PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,      # Bajo para estabilidad
    n_steps=4096,            # Largo para capturar episodios completos
    batch_size=128,
    n_epochs=10,
    gamma=0.999,             # Alto para valorar reward final
    ent_coef=0.05,           # Mas exploracion (crucial para sparse)
    gae_lambda=0.98,         # GAE alto para propagar reward
)
```

### Conversion de Agentes

Cuando un healthy es infectado:
1. Se elimina el HealthyAgent
2. Se crea un nuevo InfectedAgent en la misma posicion
3. El nuevo infected hereda las ventajas (velocidad, vision)
4. El modelo infected controla al nuevo agente

---

## Configuracion de Ventajas (EnvConfig)

```python
from src.envs import EnvConfig

config = EnvConfig(
    # ... otros parametros ...
    infected_speed=2,           # Celdas por movimiento
    infected_global_vision=True # Ver todos los healthy
)
```

---

## Compatibilidad de Modelos

**IMPORTANTE**: Los modelos entrenados con versiones anteriores NO son compatibles con la version actual.

Para usar modelos antiguos, se requiere re-entrenamiento.

---

## Troubleshooting

### "Los agentes se quedan quietos en play.py"

1. Verificar que el path de modelos es correcto:
   ```bash
   python scripts/play.py -l 3 -m models/sparse_v1
   ```

2. Verificar que los modelos existen:
   ```bash
   ls models/sparse_v1/*.zip
   ```

### "Los infected siguen sin ganar"

Con sparse rewards + ventajas (velocidad 2x, vision global), los infected deberian poder ganar. Si no:

1. Verificar que las ventajas estan activas:
   ```python
   from src.envs import EnvConfig
   config = EnvConfig()
   print(f"infected_speed: {config.infected_speed}")  # Debe ser 2
   print(f"infected_global_vision: {config.infected_global_vision}")  # Debe ser True
   ```

2. Aumentar timesteps de entrenamiento para infected

### "Error de dimension en modelo"

Si recibes error de dimension, los modelos son incompatibles. Re-entrena con:

```bash
python scripts/train_sparse.py --output-dir models/sparse_v2
```

---

## Metricas Esperadas

| Metrica | Objetivo |
|---------|----------|
| Infected Win Rate | 40-60% |
| Healthy Win Rate | 40-60% |
| Avg Steps | 200-400 |

Un balance ~50/50 indica que ambos roles han aprendido estrategias efectivas.
